import numpy as np
import time
import scipy
import random
import os
import logging
from datetime import datetime
from scipy.optimize import linprog
import warnings
import torch  # 引入 PyTorch
import torch.nn.functional as F
from tqdm import tqdm
import time

from typing import List, Tuple, Any, Optional

from utils import *

def at_find_nearest_neighbor(D: torch.Tensor, e: torch.Tensor, u: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    【步骤 2】根据当前的锚点 (e, u) 在数据集 D 中找到加权距离平方最小的最近邻点 p

    参数:
        D (torch.Tensor): 核心候选数据集，形状为 (N, d)
        e (torch.Tensor): 当前期望中心向量，形状为 (d,)
        u (torch.Tensor): 当前属性权重向量，形状为 (d,)

    返回:
        p_idx (int): 最近邻点在数据集中的索引
        p_val (torch.Tensor): 最近邻点的具体属性向量，形状为 (d,)
    """
    # 核心张量加速区：利用广播机制计算所有点到 e 的距离
    # D - e: 将形状 (d,) 的 e 广播到 (N, d) 的 D 上，同时计算每个维度的差值
    # u * (...): 将形状 (d,) 的 u 广播到 (N, d) 的差值平方上进行逐元素权重相乘
    # torch.sum(..., dim=-1): 在最后一个维度 (维度 d) 上求和，得到形状为 (N,) 的距离向量
    distances = torch.sum(u * (D - e) ** 2, dim=-1)

    # 提取最小值对应的索引
    p_idx = torch.argmin(distances).item()

    # 获取具体的点向量
    p_val = D[p_idx]

    return p_idx, p_val

def at_check_stopping_condition_LP(
        history_constraints: List[Tuple[int, int]],
        D: torch.Tensor,
        p_idx: int,
        e: torch.Tensor,
        u: torch.Tensor,
        visited_rebuttals: set,
        device: str = "cpu"
) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    【附加控制步】利用线性规划(LP)检查当前最优解 p 是否满足终止条件。
    如果不满足，则找出一个反驳点 q，并返回用于构建后续单调路径的 e_prime 和 u_prime。

    返回:
        rebuttal_q_idx (int 或 None): 找到的反驳点索引。如果为 None，说明绝对收敛。
        e_prime (torch.Tensor): LP 解出的反驳期望点
        u_prime (torch.Tensor): LP 解出的反驳权重
    """
    n, d = D.shape
    p_star = D[p_idx]

    # 获取各维度的最大值，用于修复 z <= u * E_max 的动态边界
    E_max = torch.max(D, dim=0)[0]

    # ----------------------------------------------------
    # 核心优化 1：GPU 内核矩阵化预组装 (Zero-Loop Construction)
    # ----------------------------------------------------
    if len(history_constraints) > 0:
        winners = torch.tensor([h[0] for h in history_constraints], dtype=torch.long, device=device)
        losers = torch.tensor([h[1] for h in history_constraints], dtype=torch.long, device=device)
        P_k = D[winners]
        Q_k = D[losers]

        # 约束: g(p) - g(q) <= 0
        u_coefs_H = P_k ** 2 - Q_k ** 2
        z_coefs_H = -2 * (P_k - Q_k)
        A_ub_H = torch.cat([u_coefs_H, z_coefs_H], dim=-1)
        b_ub_H = torch.full((len(history_constraints),), -LP_ACC, device=device)
    else:
        A_ub_H = torch.empty((0, 2 * d), device=device)
        b_ub_H = torch.empty((0,), device=device)

    # 边界约束: z_i - u_i <= 0
    A_ub_bounds = torch.zeros((d, 2 * d), device=device)
    A_ub_bounds[torch.arange(d), torch.arange(d)] = -1.0
    A_ub_bounds[torch.arange(d), torch.arange(d) + d] = 1.0
    b_ub_bounds = torch.zeros(d, device=device)

    # 统一合并并转移到 CPU 内存 (仅传输一次)
    A_ub = torch.cat([A_ub_H, A_ub_bounds], dim=0).cpu().numpy()
    b_ub = torch.cat([b_ub_H, b_ub_bounds], dim=0).cpu().numpy()

    # 等式约束: sum(u_i) = 1
    A_eq = np.zeros((1, 2 * d))
    A_eq[0, :d] = 1.0
    b_eq = np.array([1.0])

    # 变量边界: u_i in [0, 1], z_i in [0, 1)
    E_max_val = torch.max(D).item()
    bounds = [(0, 1)] * d + [(0, E_max_val)] * d

    # ----------------------------------------------------
    # 核心优化 2：智能排序与极速早停 (Lazy Evaluation)
    # ----------------------------------------------------
    D_np = D.cpu().numpy()
    p_star_np = p_star.cpu().numpy()

    # 预计算所有点作为 q 时的目标函数系数 C
    u_coefs_obj = D_np ** 2 - p_star_np ** 2
    z_coefs_obj = -2 * (D_np - p_star_np)
    C_all = np.concatenate([u_coefs_obj, z_coefs_obj], axis=-1)

    # 计算当前锚点下所有点到 e 的距离，由近及远排序
    current_distances = torch.sum(u * (D - e) ** 2, dim=-1)
    sorted_q_indices = torch.argsort(current_distances).cpu().numpy()

    q_idx_list = []
    e_list = []
    u_list = []

    # 按可能性从大到小依次向 SciPy 喂数据
    for q_idx in sorted_q_indices:
        if q_idx == p_idx or q_idx in visited_rebuttals:
            continue

        c = C_all[q_idx]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        # 只要找到一个目标值 < 0 的反例，立刻终止并提取射线方向
        if res.success and res.fun < -LP_ACC:
            u_prime = res.x[:d]
            z_prime = res.x[d:]

            # 核心优化 3：安全反解，防除零溢出
            u_safe = np.clip(u_prime, LP_SAFE, None)
            e_prime = z_prime / u_safe

            # 将解转回 GPU 张量并返回
            u_prime_tensor = torch.tensor(u_prime, dtype=TORCH_DTYPE, device=device)
            e_prime_tensor = torch.tensor(e_prime, dtype=TORCH_DTYPE, device=device)

            q_idx_list.append(int(q_idx))
            e_list.append(e_prime_tensor)
            u_list.append(u_prime_tensor)

            if len(q_idx_list) > 5:
                break
            # return int(q_idx), e_prime_tensor, u_prime_tensor

    # 如果遍历完毕没有任何点能击败 p_star，算法绝对收敛
    return q_idx_list, e_list, u_list


def at_construct_q_chain(D: torch.Tensor, e_prime: torch.Tensor, u: torch.Tensor, delta_u: torch.Tensor):
    """
    【步骤 3】利用纯张量下包络线（Lower Envelope）算法，构造单调最近点路径（Monotone Path）。

    参数:
        D (torch.Tensor): 核心候选数据集，形状为 (N, d)
        e (torch.Tensor): 当前锚点的期望中心向量，形状为 (d,)
        u (torch.Tensor): 当前锚点的属性权重向量，形状为 (d,)
        delta_u (torch.Tensor): 指向反驳点的权重扰动方向，形状为 (d,)

    返回:
        q_chain (List[int]): 构成单调路径的数据点索引列表，按 eta 递增顺序排列。
    """
    # ---------------------------------------------------------
    # 核心映射：将所有点转化为关于步长 eta 的直线方程 g(p, eta) = A + B * eta
    # ---------------------------------------------------------
    # A_p = \sum u_i (p_i - e_i)^2   (即 eta = 0 时的原始加权距离)
    A = torch.sum(u * (D - e_prime) ** 2, dim=-1)

    # B_p = \sum \Delta u_i (p_i - e_i)^2  (即距离随 eta 变化的斜率)
    B = torch.sum(delta_u * (D - e_prime) ** 2, dim=-1)

    # 初始状态 eta = 0 时的最优解，必然是我们步骤 2 找到的 p_idx
    curr_p = torch.argmin(A).item()
    q_chain = [curr_p]
    q_chain_eu = [(e_prime, u.clone())]

    # ---------------------------------------------------------
    # 下包络线（Lower Envelope）极速扫描
    # ---------------------------------------------------------
    while True:
        # 寻找交点：eta = (A_curr - A_i) / (B_i - B_curr)
        # 几何法则：只有斜率更小的线（B_i < B_curr），才有可能在未来的正半轴击败当前最优线
        valid_mask = B < B[curr_p]

        # 如果没有任何一条线的斜率比当前线更小，说明当前点将永远统治未来的 eta，路径终止
        if not valid_mask.any():
            break

        # A1 + B1η = A2 + B2η
        # n = (A1-A2)/(B2-B1)

        numerator = A[curr_p] - A
        denominator = B - B[curr_p]

        eta_intersections = numerator / denominator

        # 过滤机制：排除掉不符合斜率条件的非法交点，以及过去/极近的交点（使用 1e-7 防止浮点精度导致的无限原地循环）
        eta_intersections[~valid_mask] = float('inf')
        eta_intersections[eta_intersections <= ETA] = float('inf')

        # 在所有合法的未来交点中，寻找最早发生的那个交点（即最小的 eta）
        next_p = torch.argmin(eta_intersections).item()

        # 如果算出来的最小 eta 也是正无穷，说明剩下的线只在负半轴相交，未来不再有交点，路径终止
        if eta_intersections[next_p] == float('inf'):
            break

            # 权杖交接：更新当前统治者，将其加入链条
        curr_p = next_p
        q_chain.append(curr_p)
        q_chain_eu.append((e_prime, u+eta_intersections[next_p]*delta_u))

    return q_chain, q_chain_eu

def at_interactive_binary_search(
        D: torch.Tensor,
        q_chain: List[int],
        oracle_e: torch.Tensor,
        oracle_u: torch.Tensor,
        history_constraints: List[Tuple[int, int]]
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    【步骤 4】对单调路径 q_chain 进行二分查找，通过模拟用户交互收集偏好约束 H。

    参数:
        D (torch.Tensor): 数据集 (N, d)
        q_chain (List[int]): 步骤 3 提取的单调最近点路径
        oracle_e, oracle_u: 用户的真实偏好（用于模拟真实用户的判断）
        history_constraints: 历史约束集合 H

    返回:
        final_q_idx (int): 沿这条射线二分查找最终收敛的那个用户最喜欢的点
        queries_used (int): 本轮查找消耗的提问次数
        history_constraints (List): 追加了本轮新约束的 H 集合
    """
    queries_used = 0

    # 当链条中剩余多于 1 个候选点时，持续进行二分查找
    while len(q_chain) > 1:
        # 寻找最中间的两个相邻候选点
        mid = len(q_chain) // 2
        p_left = q_chain[mid - 1]
        p_right = q_chain[mid]

        # ----------------------------------------------------
        # 模拟用户判断 (Oracle)
        # 真实场景下，这里应替换为向用户前端发送问题并等待接收点击反馈
        # ----------------------------------------------------
        dist_left = torch.sum(oracle_u * (D[p_left] - oracle_e) ** 2)
        dist_right = torch.sum(oracle_u * (D[p_right] - oracle_e) ** 2)
        user_prefers_left = dist_left < dist_right

        queries_used += 1


        # ----------------------------------------------------
        # 二分剪枝与约束收集
        # ----------------------------------------------------
        if user_prefers_left:
            # 用户觉得左边的点更好，说明真实理想点对应的 eta 必定 <= 临界点
            # 右半部分的链条永远不可能是最优解，整体剪裁掉
            history_constraints.append((p_left, p_right))
            q_chain = q_chain[:mid]
        else:
            # 用户觉得右边的点更好，说明真实理想点对应的 eta 必定 >= 临界点
            # 左半部分的链条被淘汰，整体剪裁掉
            history_constraints.append((p_right, p_left))
            q_chain = q_chain[mid:]

    # 最终链条必定会收束到唯一一个点（局部最优解）
    final_q_idx = q_chain[0] if len(q_chain) > 0 else -1
    # ==

    return final_q_idx, queries_used, history_constraints


def at_alternating_learning(
        D: torch.Tensor,
        history_constraints: List[Tuple[int, int]],
        current_e: torch.Tensor,
        current_u: torch.Tensor,
        max_iter: int = 500,
        init_lr: float = 0.5,
        device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【步骤 5】基于最新的历史偏好约束 H，通过计算似然函数与回溯线搜索，交替优化锚点。
    """
    e, u = current_e.clone(), current_u.clone()

    if len(history_constraints) == 0:
        return e, u

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]

    p_indices = torch.tensor([h[0] for h in history_constraints], dtype=torch.long, device=device)
    q_indices = torch.tensor([h[1] for h in history_constraints], dtype=torch.long, device=device)
    P = D[p_indices]
    Q = D[q_indices]

    def compute_likelihood_and_constraints(e_test: torch.Tensor, u_test: torch.Tensor) -> Tuple[float, bool]:
        """核心组件：计算当前参数的 Likelihood (负对数似然) 以及是否满足 H 约束"""
        dist_P = torch.sum(u_test * (P - e_test) ** 2, dim=-1)
        dist_Q = torch.sum(u_test * (Q - e_test) ** 2, dim=-1)
        delta = dist_P - dist_Q

        # 1. 计算 Likelihood = sum(log(1 + exp(g(p) - g(q))))
        # 使用 softplus 保证数值稳定性，避免 delta 过大时出现 NaN
        L = torch.sum(F.softplus(delta)).item()

        # 2. 校验所有的历史偏好是否绝对满足 (期望 delta < 0)
        is_valid = torch.all(delta < 0).item()

        return L, is_valid

    for i in range(max_iter):
        e_prev, u_prev = e.clone(), u.clone()

        # ------------------------------------------------
        # 1. 固定 e，计算 Likelihood 偏导并退火更新 u
        # ------------------------------------------------
        L_prev, _ = compute_likelihood_and_constraints(e, u)

        term_p_u = (P - e) ** 2
        term_q_u = (Q - e) ** 2
        delta_u = torch.sum(u * term_p_u, dim=-1) - torch.sum(u * term_q_u, dim=-1)

        # L 对 delta 的偏导数是 sigmoid
        sig_u = torch.sigmoid(delta_u).unsqueeze(-1)
        # 链式法则：将 Likelihood 对 u 求偏导
        grad_u = torch.sum(sig_u * (term_p_u - term_q_u), dim=0)

        lr = init_lr

        while lr > 1e-5:
            u_next = torch.clamp(u - lr * grad_u, 0.001, 1.0)
            u_next = u_next / torch.sum(u_next)

            # 计算更新后的 Likelihood 和合法性
            L_next, is_valid = compute_likelihood_and_constraints(e, u_next)

            # 退火核心：只有当 Likelihood 真实下降，且满足约束时，才接受更新！
            if L_next < L_prev:
                u = u_next
                break
            # 否则，学习率砍半，重新试探
            lr *= 0.5

            # ------------------------------------------------
        # 2. 固定 u，计算 Likelihood 偏导并退火更新 e
        # ------------------------------------------------
        L_prev, _ = compute_likelihood_and_constraints(e, u)

        delta_e = torch.sum(u * (P - e) ** 2, dim=-1) - torch.sum(u * (Q - e) ** 2, dim=-1)
        sig_e = torch.sigmoid(delta_e).unsqueeze(-1)

        # 链式法则：将 Likelihood 对 e 求偏导 (化简后为 2*u*(Q-P))
        grad_e = torch.sum(sig_e * (2 * u * (Q - P)), dim=0)

        lr = init_lr
        while lr > 1e-5:
            e_next = torch.clamp(e - lr * grad_e, E_min, E_max)

            L_next, is_valid = compute_likelihood_and_constraints(e_next, u)

            if L_next < L_prev:
                e = e_next
                break
            lr *= 0.5
        # print("After +++++++++++++++++++++++++++++++++++++")
        # print(f"e:{e}, u:{u}")
        # print("============== END OF ROUND =================")
            # 极小幅度早停判定
        if i > 20 and torch.linalg.norm(e - e_prev) < 1e-4 and torch.linalg.norm(u - u_prev) < 1e-4:
            # print(f"============== BREAK AT {i} =================")
            break



    return e, u


def anchor_tracking_algorithm(D, true_e, true_u, max_round=500):
    # [AT 主框架] 算法启动
    device = D.device
    d = D.shape[1]

    # ----------------------------------------------------
    # 步骤 1: (Init) 随机生成一组 e 和 u
    # ----------------------------------------------------
    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]
    e = E_min + torch.rand(d, device=device) * (E_max - E_min)

    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))
    u = dirichlet_dist.sample()

    # 初始化历史约束集合 H (用于存储用户对特定元组对的反馈)
    H = []
    total_queries_used = 0

    rounds = 0

    visited_rebuttals = set()
    previous_p_idx = -1
    previous_e = e
    previous_u = u
    previous_qchain = []


    pbar = tqdm(total=max_round)
    while rounds < max_round:

        # ----------------------------------------------------
        # 步骤 2: 根据这一组 e 和 u 找到最近邻的 p
        # ----------------------------------------------------
        p_idx, p_val = at_find_nearest_neighbor(D, e, u)

        _rebuttal_q_idx, _e_prime, _u_prime = at_check_stopping_condition_LP(H, D, p_idx, e, u, visited_rebuttals, device=device)
        if len(_rebuttal_q_idx) == 0:
            logging.info("Ending-LP")
            # distance = get_weighted_distance_sq(D, true_e, true_u)
            # true_best_idx = torch.argmin(distance).item()
            # if p_idx != true_best_idx:
            #     visited_rebuttals = set(range(0, 20000)) - {true_best_idx}
            #     _rebuttal_q_idx, _e_prime, _u_prime = at_check_stopping_condition_LP(H, D, p_idx, e, u,
            #                                                                          visited_rebuttals, device=device)
            pbar.close()
            return p_idx, rounds, total_queries_used

        rebuttal_q_idx, e_prime, u_prime = _rebuttal_q_idx[0], _e_prime[0], _u_prime[0]


        delta_u = u_prime - u
        q_chain, q_chain_eu = at_construct_q_chain(D, e_prime, u, delta_u)

        # P不好-跳转
        p = torch.sum(true_u * (D[p_idx] - true_e) ** 2)
        q = torch.sum(true_u * (D[rebuttal_q_idx] - true_e) ** 2)
        user_prefers_p = p < q
        total_queries_used += 1
        if not user_prefers_p:
            # print(f"touch0:P:{p_idx}-->Q:{rebuttal_q_idx}")
            if (rebuttal_q_idx, p_idx) in H:
                visited_rebuttals.add(p_idx)
                # visited_rebuttals.add(rebuttal_q_idx)
            else:
                H.append((rebuttal_q_idx, p_idx))
            # e, u = find_eu_making_q_nearest(D, rebuttal_q_idx, device=device)
            # q_idx, _ = at_find_nearest_neighbor(D, e, u)
            e = e_prime
            u = u_prime
            rounds += 1
            pbar.update(1)
        else:
            # P好-不重复-走后续流程
            # print("touched1.5")
            H.append((p_idx, rebuttal_q_idx))
            if q_chain != previous_qchain or p_idx != previous_p_idx or (not torch.equal(e, previous_e)) or (not torch.equal(u, previous_u)):
                # if  q_chain != previous_qchain:
                #     print(1, end="")
                # if p_idx != previous_p_idx:
                #     print(2, end="")
                # if (not torch.equal(e, previous_e)):
                #     print(3, end="")
                # if (not torch.equal(u, previous_u)):
                #     print(4, end="")
                # print("")
                visited_rebuttals.clear()
                previous_p_idx = p_idx
                previous_e = e
                previous_u = u
                previous_qchain = q_chain

                final_q, queries, H = at_interactive_binary_search(
                    D=D,
                    q_chain=q_chain,
                    oracle_e=true_e,
                    oracle_u=true_u,
                    history_constraints=H
                )
                total_queries_used += queries
                e, u = at_alternating_learning(D, H, e, u, device=device)
                rounds += 1
                pbar.update(1)
            else:
                # P好，重复-锦标赛
                print("touched2")
                best = p_idx
                for i in range(len(_rebuttal_q_idx)):
                    visited_rebuttals.add(_rebuttal_q_idx[i])
                    P = torch.sum(true_u * (D[best] - true_e) ** 2)
                    Q = torch.sum(true_u * (D[_rebuttal_q_idx[i]] - true_e) ** 2)
                    total_queries_used += 1
                    if Q < P:
                        best = _rebuttal_q_idx[i]
                        e_prime = _e_prime[i]
                        u_prime = _u_prime[i]
                        H.append((_rebuttal_q_idx[i], best))
                    else:
                        H.append((best, _rebuttal_q_idx[i]))
                    # P不是最优，跳转
                if best != p_idx:
                    e = e_prime
                    u = u_prime
                    rounds += 1
                    pbar.update(1)
                else:
                    print("touched3")
                    final_q, queries, H = at_interactive_binary_search(
                        D=D,
                        q_chain=q_chain,
                        oracle_e=true_e,
                        oracle_u=true_u,
                        history_constraints=H
                    )
                    total_queries_used += queries
                    e, u = at_alternating_learning(D, H, e, u, device=device)
                    rounds += 1
                    pbar.update(1)




    # 超过 max_queries 未能收敛,返回当前结果
    # print(f"touched:{touched}")
    pbar.close()
    logging.info("touched3")
    return rebuttal_q_idx, rounds, total_queries_used



def main(args):
    global TORCH_DTYPE,  LP_ACC, LP_SAFE, ETA
    if args.dtype == "float64":
        TORCH_DTYPE = torch.float64
        LP_ACC = 1e-08
        LP_SAFE = 1e-09
        ETA = 1e-10
    else: # Default is float32. For 2D dataset, should be torch.float64 to ensure 100% accuracy
        TORCH_DTYPE = torch.float32
        LP_ACC = 1e-06
        LP_SAFE = 1e-07
        ETA = 1e-07

    setup_logging()  # 初始化日志    
    filepath = args.data
    D_np = load_dataset(filepath)
    if D_np is None:
        Exception("Wrong dataset path.")

    if args.data_size < len(D_np):
        D_np = D_np[:args.data_size]

    num_experiments = args.rounds

    # 自动识别并配置 GPU / CPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"数据集: {filepath}")
    logging.info(f"当前运行设备: {device}")

    # 将数据集加载为 PyTorch Tensor 并推送至指定设备
    D = torch.tensor(D_np, dtype=TORCH_DTYPE, device=device)
    n, d = D.shape
    logging.info(f"成功加载数据集: {n} 条数据, {d} 个维度。")

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]

    rs_results, at_results = [], []
    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))

    logging.info(f"开始执行 {num_experiments} 次对比实验...")

    for i in range(num_experiments):
        logging.info(f"\n--- 实验 {i + 1}/{num_experiments} ---")

        true_u = dirichlet_dist.sample()
        true_e = E_min + torch.rand(d, device=device) * (E_max - E_min)

        distance = get_weighted_distance_sq(D, true_e, true_u)
        true_best_idx = torch.argmin(distance).item()
        best_distance = torch.min(distance).item()
        logging.info(f"  [Oracle] 用户的真实最优数据点索引为: {true_best_idx}")


        start_time = time.time()
        at_best_idx, rounds, at_queries = anchor_tracking_algorithm(D, true_e, true_u)
        at_time = time.time() - start_time
        at_success = (at_best_idx == true_best_idx)

        # --- 新增：AT 评估指标计算 ---
        if at_best_idx != -1:
            # 获取排名 (注意：PyTorch argsort 后找 index 的操作)
            sorted_indices = torch.argsort(distance)
            rank = (sorted_indices == at_best_idx).nonzero(as_tuple=True)[0][0].item()
            # 计算距离偏差 (加 1e-9 防止分母为 0)
            dis_bias = (get_weighted_distance_sq(D[at_best_idx], true_e, true_u).item() - best_distance) / (
                        best_distance + 1e-9)
        else:
            rank = n - 1
            dis_bias = float('inf')

        at_results.append({
            'queries': at_queries,
            'success': at_success,
            'time': at_time,
            'dis_bias': dis_bias,
            'rank': rank
        })

        logging.info(
            f"  [AT 算法] 返回点: {at_best_idx} | 排名: {rank + 1}/{n} | 偏差: {dis_bias:.4f} | 运行次数: {rounds} |提问次数: {at_queries} | 命中: {at_success}")

    # 5. 统计报告
    logging.info("\n" + "=" * 40)
    logging.info(f"实验结果统计 ( {num_experiments}次平均)")
    logging.info("=" * 40)

    at_valid_results = [r for r in at_results if r['rank'] != (n - 1)]

    if len(at_valid_results) > 0:
        at_avg_queries = np.mean([r['queries'] for r in at_valid_results])
        at_rank = np.mean([(r['rank'] + 1) for r in at_valid_results])
        at_bias = np.mean([r['dis_bias'] for r in at_valid_results])
    else:
        at_avg_queries = at_rank = at_bias = float('nan')

    at_success_rate = np.mean([r['success'] for r in at_results])
    at_top5 = np.sum([1 for r in at_results if r['rank'] < 5])  # rank 是 0 索引，< 5 即 Top-5
    at_fail = np.sum([1 for r in at_results if r['rank'] == (n - 1)])
    at_avg_time = np.mean([r['time'] for r in at_results])

    logging.info(f"[Anchor Tracking] 命中率: {at_success_rate:.4f}")
    logging.info(f"[Anchor Tracking] Top-5 命中率: {at_top5}/{num_experiments}")
    logging.info(f"[Anchor Tracking] 平均排位: {at_rank:.1f}/{n}")
    logging.info(f"[Anchor Tracking] 平均偏差: {at_bias:.4f}")
    logging.info(f"[Anchor Tracking] 平均提问次数: {at_avg_queries:.1f}")
    logging.info(f"[Anchor Tracking] 平均耗时: {at_avg_time:.4f} 秒")
    logging.info(f"[Anchor Tracking] 失败次数: {at_fail}/{num_experiments}")



if __name__ == "__main__":
    main()