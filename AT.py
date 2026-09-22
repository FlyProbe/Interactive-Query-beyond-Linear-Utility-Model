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
import json

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
    如果不满足，则找出【第一个】反驳点 q（按距离由近及远），并返回。
    去除了冗余的 List 列表包装，直接返回单个反驳点及参数。
    """
    n, d = D.shape
    p_star = D[p_idx]

    E_max = torch.max(D, dim=0)[0]

    # --- 1. GPU 内核矩阵化预组装约束 ---
    if len(history_constraints) > 0:
        winners = torch.tensor([h[0] for h in history_constraints], dtype=torch.long, device=device)
        losers = torch.tensor([h[1] for h in history_constraints], dtype=torch.long, device=device)
        P_k = D[winners]
        Q_k = D[losers]

        u_coefs_H = P_k ** 2 - Q_k ** 2
        z_coefs_H = -2 * (P_k - Q_k)
        A_ub_H = torch.cat([u_coefs_H, z_coefs_H], dim=-1)
        b_ub_H = torch.full((len(history_constraints),), -LP_ACC, device=device)
    else:
        A_ub_H = torch.empty((0, 2 * d), device=device)
        b_ub_H = torch.empty((0,), device=device)

    A_ub_bounds = torch.zeros((d, 2 * d), device=device)
    A_ub_bounds[torch.arange(d), torch.arange(d)] = -1.0
    A_ub_bounds[torch.arange(d), torch.arange(d) + d] = 1.0
    b_ub_bounds = torch.zeros(d, device=device)

    A_ub = torch.cat([A_ub_H, A_ub_bounds], dim=0).cpu().numpy()
    b_ub = torch.cat([b_ub_H, b_ub_bounds], dim=0).cpu().numpy()

    A_eq = np.zeros((1, 2 * d))
    A_eq[0, :d] = 1.0
    b_eq = np.array([1.0])

    E_max_val = torch.max(D).item()
    bounds = [(0, 1)] * d + [(0, E_max_val)] * d

    # --- 2. 智能排序与极速早停 (Lazy Evaluation) ---
    D_np = D.cpu().numpy()
    p_star_np = p_star.cpu().numpy()

    u_coefs_obj = D_np ** 2 - p_star_np ** 2
    z_coefs_obj = -2 * (D_np - p_star_np)
    C_all = np.concatenate([u_coefs_obj, z_coefs_obj], axis=-1)

    current_distances = torch.sum(u * (D - e) ** 2, dim=-1)
    sorted_q_indices = torch.argsort(current_distances).cpu().numpy()

    for q_idx in sorted_q_indices:
        q_idx = int(q_idx)
        if q_idx == p_idx or q_idx in visited_rebuttals:
            continue

        c = C_all[q_idx]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success and res.fun < -LP_ACC:
            u_prime = res.x[:d]
            z_prime = res.x[d:]

            u_safe = np.clip(u_prime, LP_SAFE, None)
            e_prime = z_prime / u_safe

            u_prime_tensor = torch.tensor(u_prime, dtype=TORCH_DTYPE, device=device)
            e_prime_tensor = torch.tensor(e_prime, dtype=TORCH_DTYPE, device=device)

            # 【去除 List，直接触发早停并返回单一结果】
            return q_idx, e_prime_tensor, u_prime_tensor

    return None, None, None


def at_construct_q_chain_alternating(D: torch.Tensor, e: torch.Tensor, u: torch.Tensor,
                                     e_prime: torch.Tensor, u_prime: torch.Tensor,
                                     vary_u: bool):
    """
    【步骤 3】支持交替策略的下包络线（Lower Envelope）算法构造单调最近点路径。

    参数:
        vary_u (bool): True 表示变动 u 固定 e；False 表示变动 e 固定 u。
    """
    if vary_u:
        # ---------------------------------------------------------
        # 模式 1：固定 e_prime，变动 u (u + eta * delta_u)
        # ---------------------------------------------------------
        delta_u = u_prime - u
        # A_p = \sum u_i (p_i - e_prime_i)^2
        A = torch.sum(u * (D - e_prime) ** 2, dim=-1)
        # B_p = \sum \Delta u_i (p_i - e_prime_i)^2
        B = torch.sum(delta_u * (D - e_prime) ** 2, dim=-1)

        curr_p = torch.argmin(A).item()
        q_chain = [curr_p]
        q_chain_eu = [(e_prime, u.clone())]

        while True:
            valid_mask = B < B[curr_p]
            if not valid_mask.any():
                break

            numerator = A[curr_p] - A
            denominator = B - B[curr_p]
            eta_intersections = numerator / denominator

            eta_intersections[~valid_mask] = float('inf')
            eta_intersections[eta_intersections <= ETA] = float('inf')

            next_p = torch.argmin(eta_intersections).item()
            if eta_intersections[next_p] == float('inf'):
                break

            curr_p = next_p
            q_chain.append(curr_p)
            q_chain_eu.append((e_prime, u + eta_intersections[next_p] * delta_u))

    else:
        # ---------------------------------------------------------
        # 模式 2：固定 u_prime，变动 e (e + eta * delta_e)
        # ---------------------------------------------------------
        delta_e = e_prime - e
        # A_p = \sum u_prime_i (p_i - e_i)^2
        A = torch.sum(u_prime * (D - e) ** 2, dim=-1)
        # B_p = -2 * \sum u_prime_i (p_i - e_i) * \Delta e_i
        # 注意：此处二阶项 eta^2 在求交点差值时已抵消，所以斜率 B 仅保留一阶交叉项
        B = -2.0 * torch.sum(u_prime * D * delta_e, dim=-1)

        curr_p = torch.argmin(A).item()
        q_chain = [curr_p]
        q_chain_eu = [(e.clone(), u_prime)]

        while True:
            valid_mask = B < B[curr_p]
            if not valid_mask.any():
                break

            numerator = A[curr_p] - A
            denominator = B - B[curr_p]
            eta_intersections = numerator / denominator

            eta_intersections[~valid_mask] = float('inf')
            eta_intersections[eta_intersections <= ETA] = float('inf')

            next_p = torch.argmin(eta_intersections).item()
            if eta_intersections[next_p] == float('inf'):
                break

            curr_p = next_p
            q_chain.append(curr_p)
            q_chain_eu.append((e + eta_intersections[next_p] * delta_e, u_prime))

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
        DATA["Time"].append(time.time() - DATA["Time"][0])


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
    修改点：
    1. 恢复 is_valid 的硬拦截，严格保证锚点不越出可行域。
    2. 将梯度计算由 sum 改为 mean，解耦梯度尺度与 H 的大小。
    3. 放宽免检期并收紧早停容差，榨干边界下降空间。
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

        sig_u = torch.sigmoid(delta_u).unsqueeze(-1)

        grad_u = torch.mean(sig_u * (term_p_u - term_q_u), dim=0)

        lr = init_lr

        while lr > 1e-5:
            u_next = torch.clamp(u - lr * grad_u, 0.001, 1.0)
            u_next = u_next / torch.sum(u_next)

            L_next, is_valid = compute_likelihood_and_constraints(e, u_next)

            # 严格遵守 Bradley-Terry 模型要求锚点解释所有历史反馈的前提
            if L_next < L_prev and is_valid:
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

        grad_e = torch.mean(sig_e * (2 * u * (Q - P)), dim=0)

        lr = init_lr
        while lr > 1e-5:
            e_next = torch.clamp(e - lr * grad_e, E_min, E_max)

            L_next, is_valid = compute_likelihood_and_constraints(e_next, u)

            if L_next < L_prev and is_valid:
                e = e_next
                break
            lr *= 0.5

        # 给予均值化梯度更长的探索窗口，逼迫算法利用极微小的合法梯度贴墙滑动
        if i > 50 and torch.linalg.norm(e - e_prev) < 1e-5 and torch.linalg.norm(u - u_prev) < 1e-5:
            break


    return e, u


def anchor_tracking_algorithm(D, true_e, true_u, max_round=500):
    device = D.device
    d = D.shape[1]

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]
    e = E_min + torch.rand(d, device=device) * (E_max - E_min)

    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))
    u = dirichlet_dist.sample()

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

        p_idx, p_val = at_find_nearest_neighbor(D, e, u)

        # 【直接接收单一返回值】
        rebuttal_q_idx, e_prime, u_prime = at_check_stopping_condition_LP(H, D, p_idx, e, u, visited_rebuttals,
                                                                          device=device)

        # 判定是否收敛
        if rebuttal_q_idx is None:
            logging.info("Ending-LP")
            pbar.close()
            return p_idx, rounds, total_queries_used

        vary_u = (rounds % 2 == 0)

        q_chain, q_chain_eu = at_construct_q_chain_alternating(
            D=D, e=e, u=u, e_prime=e_prime, u_prime=u_prime, vary_u=vary_u
        )

        p = torch.sum(true_u * (D[p_idx] - true_e) ** 2)
        q = torch.sum(true_u * (D[rebuttal_q_idx] - true_e) ** 2)
        user_prefers_p = p < q
        total_queries_used += 1

        if not user_prefers_p:
            # P不好-跳转
            if (rebuttal_q_idx, p_idx) in H:
                visited_rebuttals.add(p_idx)
            else:
                H.append((rebuttal_q_idx, p_idx))

            e = e_prime
            u = u_prime
            DATA["E-U Route"].append([e.tolist(), u.tolist()])
            DATA["Time"].append(time.time() - DATA["Time"][0])
            rounds += 1
            pbar.update(1)
        else:
            # P好-不重复-走后续流程
            H.append((p_idx, rebuttal_q_idx))
            if q_chain != previous_qchain or p_idx != previous_p_idx or (not torch.equal(e, previous_e)) or (
            not torch.equal(u, previous_u)):
                visited_rebuttals.clear()
                previous_p_idx = p_idx
                previous_e = e
                previous_u = u
                previous_qchain = q_chain

                final_q, queries, H = at_interactive_binary_search(
                    D=D, q_chain=q_chain, oracle_e=true_e, oracle_u=true_u, history_constraints=H
                )
                total_queries_used += queries
                e, u = at_alternating_learning(D, H, e, u, device=device)
                DATA["E-U Route"].extend([[e.tolist(), u.tolist()]]* queries)


                rounds += 1
                pbar.update(1)
            else:
                # P好，重复-锦标赛 【去除 for 循环，逻辑化简】
                print("touched2")
                visited_rebuttals.add(rebuttal_q_idx)

                best = p_idx
                P_dist = torch.sum(true_u * (D[best] - true_e) ** 2)
                Q_dist = torch.sum(true_u * (D[rebuttal_q_idx] - true_e) ** 2)
                total_queries_used += 1

                if Q_dist < P_dist:
                    best = rebuttal_q_idx
                    H.append((rebuttal_q_idx, best))
                else:
                    H.append((best, rebuttal_q_idx))

                # P不是最优，跳转
                if best != p_idx:
                    e = e_prime
                    u = u_prime
                    DATA["E-U Route"].append([e.tolist(), u.tolist()])
                    DATA["Time"].append(time.time() - DATA["Time"][0])

                    rounds += 1
                    pbar.update(1)
                else:
                    print("touched3")
                    final_q, queries, H = at_interactive_binary_search(
                        D=D, q_chain=q_chain, oracle_e=true_e, oracle_u=true_u, history_constraints=H
                    )
                    total_queries_used += queries
                    e, u = at_alternating_learning(D, H, e, u, device=device)
                    DATA["E-U Route"].extend([[e.tolist(), u.tolist()]]* queries)

                    rounds += 1
                    pbar.update(1)

    pbar.close()
    logging.info("touched3")
    return rebuttal_q_idx, rounds, total_queries_used



def main(args):
    global TORCH_DTYPE,  LP_ACC, LP_SAFE, ETA, EPS
    global DATA
    DEBUG = True
    ROUTE_PATH = "4d10kAT_deltaEU"
    EPS = args.eps
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

        DATA = {
            "Best P": [],
            "E-U Route": [],
            "True E-U": [],
            "Time": []
        }

        true_u = dirichlet_dist.sample()
        true_e = E_min + torch.rand(d, device=device) * (E_max - E_min)

        DATA["True E-U"] = [true_e.tolist(), true_e.tolist()]

        distance = get_weighted_distance_sq(D, true_e, true_u)
        true_best_idx = torch.argmin(distance).item()
        best_distance = torch.min(distance).item()
        logging.info(f"  [Oracle] 用户的真实最优数据点索引为: {true_best_idx}")

        DATA["Best P"] = D[true_best_idx].tolist()

        start_time = time.time()

        DATA["Time"].append(start_time)

        at_best_idx, rounds, at_queries = anchor_tracking_algorithm(D, true_e, true_u)
        at_time = time.time() - start_time
        # at_success = (at_best_idx == true_best_idx)

        if at_best_idx != -1:
            # 获取 AT 算法找到的点的实际加权距离
            at_distance = get_weighted_distance_sq(D[at_best_idx], true_e, true_u).item()
            # 判断距离是否在允许的误差范围内：d <= d* * (1 + EPS)
            at_success = (at_distance <= best_distance * (1 + EPS))
        else:
            # 如果算法未返回有效点，则直接判定为失败
            at_distance = float('inf')
            at_success = False



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

        DATA["Time"][0] = 0
        if DEBUG:
            with open(os.path.join(ROUTE_PATH, f"eu_route_{i}.json"), "w") as f:
                json.dump(DATA, f)

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