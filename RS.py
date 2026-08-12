import numpy as np
import time
import scipy
import os
import logging
from datetime import datetime
from scipy.optimize import linprog
import warnings
import torch  # 引入 PyTorch

from tqdm import tqdm
import time

from utils import *

def range_shrink_algorithm(D, true_e, true_u, initial_N=5000, max_queries=200, eps=0.0):
    """
    SMC + Hit-and-Run (贴边游走) 架构 + 容差交集投票
    """
    device = D.device
    n, d = D.shape
    H = []

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]

    e_samples = E_min + torch.rand(initial_N, d, device=device) * (E_max - E_min)
    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))
    u_samples = dirichlet_dist.sample((initial_N,))

    rounds = 0
    from tqdm import tqdm
    tbar = tqdm(total=max_queries, desc="Hit-and-Run Interacting")

    while rounds < max_queries:
        rounds += 1
        tbar.update(1)

        # 1. 产生交互与获取反馈
        perm = torch.randperm(n, device=device)
        p_idx, q_idx = perm[0].item(), perm[1].item()

        if simulate_user_oracle(p_idx, q_idx, D, true_e, true_u):
            winner_idx, loser_idx = p_idx, q_idx
        else:
            winner_idx, loser_idx = q_idx, p_idx

        H.append((winner_idx, loser_idx))

        # 2. 筛选幸存者
        valid_mask = torch.ones(initial_N, dtype=torch.bool, device=device)
        for w_idx, l_idx in H:
            dist_w = get_weighted_distance_sq(D[w_idx], e_samples, u_samples)
            dist_l = get_weighted_distance_sq(D[l_idx], e_samples, u_samples)
            valid_mask &= (dist_w < dist_l)

        num_survivors = valid_mask.sum().item()
        num_dead = initial_N - num_survivors

        # 3. 锦标赛兜底 (全军覆没情况)
        if num_survivors == 0:
            H.pop()
            prev_votes = []
            chunk_size = 5000
            for i in range(0, initial_N, chunk_size):
                e_chunk = e_samples[i: i + chunk_size].unsqueeze(1)
                u_chunk = u_samples[i: i + chunk_size].unsqueeze(1)
                dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
                prev_votes.append(torch.argmin(dist_chunk, dim=-1))

            unique_candidates = torch.unique(torch.cat(prev_votes))
            best_idx = unique_candidates[0].item()
            for i in range(1, len(unique_candidates)):
                competitor_idx = unique_candidates[i].item()
                rounds += 1
                tbar.update(1)
                if not simulate_user_oracle(best_idx, competitor_idx, D, true_e, true_u):
                    best_idx = competitor_idx
            tbar.close()
            # 统一返回列表格式
            return [best_idx], rounds

        # ---------------------------------------------------------
        # 4. 存活者容差交集投票判定 (核心修改区)
        # ---------------------------------------------------------
        survivors_e = e_samples[valid_mask]
        survivors_u = u_samples[valid_mask]

        # 初始化交集掩码，初始认为所有点都在交集中
        intersection_mask = torch.ones(n, dtype=torch.bool, device=device)
        chunk_size = 5000

        for i in range(0, num_survivors, chunk_size):
            e_chunk = survivors_e[i: i + chunk_size].unsqueeze(1)
            u_chunk = survivors_u[i: i + chunk_size].unsqueeze(1)

            # 计算距离平方
            dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)

            # 寻找每个粒子到数据集的最短距离的平方 d^2
            min_dist_sq = torch.min(dist_chunk, dim=-1, keepdim=True)[0]

            # 宽容度阈值: d^2 * (1 + eps)^2
            threshold_sq = min_dist_sq * ((1.0 + eps) ** 2)

            # (2) 投出距离 d(1+eps) 范围内的所有点
            vote_mask = dist_chunk <= threshold_sq

            # 计算当前 chunk 内部所有粒子的共识交集
            chunk_intersection = torch.all(vote_mask, dim=0)
            intersection_mask &= chunk_intersection

        # 提取交集内的所有元素
        intersection_indices = torch.nonzero(intersection_mask).squeeze(-1)

        # 如果所有存活粒子的返回列表有交集，则直接返回交集内的所有元素
        if len(intersection_indices) > 0:
            tbar.close()
            return intersection_indices.tolist(), rounds - 1

        elif rounds == max_queries:
            # 如果到达 max_queries 仍无交集，退化为锦标赛或返回得票最多的集合
            # 为保持严谨，这里取传统最近邻模式的众数作为 Fallback
            votes = []
            for i in range(0, num_survivors, chunk_size):
                e_chunk = survivors_e[i: i + chunk_size].unsqueeze(1)
                u_chunk = survivors_u[i: i + chunk_size].unsqueeze(1)
                dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
                votes.append(torch.argmin(dist_chunk, dim=-1))

            unique_votes = torch.unique(torch.cat(votes))
            best_idx = unique_votes[0].item()
            for i in range(1, len(unique_votes)):
                competitor_idx = unique_votes[i].item()
                rounds += 1
                tbar.update(1)
                if not simulate_user_oracle(best_idx, competitor_idx, D, true_e, true_u):
                    best_idx = competitor_idx
            tbar.close()
            return [best_idx], rounds

        # ---------------------------------------------------------
        # 5. Hit-and-Run 定向贴边补齐阵亡粒子
        # ---------------------------------------------------------
        if num_dead > 0:
            clone_idx = torch.randint(0, num_survivors, (num_dead,), device=device)
            c_e = survivors_e[clone_idx]
            c_u = survivors_u[clone_idx]

            Delta_e = torch.randn_like(c_e)
            Delta_e = Delta_e / torch.norm(Delta_e, dim=-1, keepdim=True)

            Delta_u = torch.randn_like(c_u)
            Delta_u = Delta_u - Delta_u.mean(dim=-1, keepdim=True)
            Delta_u = Delta_u / (torch.norm(Delta_u, dim=-1, keepdim=True) + 1e-9)

            t_e_min = torch.where(Delta_e < 0, (E_min - c_e) / Delta_e, float('inf'))
            t_e_max = torch.where(Delta_e > 0, (E_max - c_e) / Delta_e, float('inf'))
            t_e = torch.minimum(t_e_min.min(dim=-1)[0], t_e_max.min(dim=-1)[0])

            t_u_min = torch.where(Delta_u < 0, (0.001 - c_u) / Delta_u, float('inf'))
            t_u_max = torch.where(Delta_u > 0, (1.0 - c_u) / Delta_u, float('inf'))
            t_u = torch.minimum(t_u_min.min(dim=-1)[0], t_u_max.min(dim=-1)[0])

            t_phys = torch.minimum(t_e, t_u)
            t_logic = torch.full((num_dead,), float('inf'), device=device)

            for w_idx, l_idx in H:
                p = D[w_idx]
                q = D[l_idx]

                C = (p - c_e) ** 2 - (q - c_e) ** 2
                D_mat = -2 * Delta_e * (p - q)

                c = torch.sum(c_u * C, dim=-1)
                b = torch.sum(c_u * D_mat + Delta_u * C, dim=-1)
                a = torch.sum(Delta_u * D_mat, dim=-1)

                disc = b ** 2 - 4 * a * c
                t1 = torch.where(a != 0, (-b + torch.sqrt(torch.clamp(disc, min=0))) / (2 * a + 1e-12),
                                 -c / (b + 1e-12))
                t2 = torch.where(a != 0, (-b - torch.sqrt(torch.clamp(disc, min=0))) / (2 * a + 1e-12), float('inf'))

                t1 = torch.where((disc >= 0) & (t1 > 1e-6), t1, float('inf'))
                t2 = torch.where((disc >= 0) & (t2 > 1e-6), t2, float('inf'))

                t_h_min = torch.minimum(t1, t2)
                t_logic = torch.minimum(t_logic, t_h_min)

            t_max = torch.minimum(t_phys, t_logic)
            rand_ratios = torch.rand(num_dead, device=device)
            t_step = rand_ratios * 0.999 * t_max

            new_e = c_e + t_step.unsqueeze(-1) * Delta_e
            new_u = c_u + t_step.unsqueeze(-1) * Delta_u

            new_e = torch.max(torch.min(new_e, E_max), E_min)
            new_u = torch.clamp(new_u, 0.001, 1.0)
            new_u = new_u / new_u.sum(dim=-1, keepdim=True)

            e_samples[~valid_mask] = new_e
            u_samples[~valid_mask] = new_u

    tbar.close()

    # 达到 max_queries 未收敛兜底
    votes = []
    for i in range(0, initial_N, 5000):
        e_chunk = e_samples[i: i + 5000].unsqueeze(1)
        u_chunk = u_samples[i: i + 5000].unsqueeze(1)
        dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
        votes.append(torch.argmin(dist_chunk, dim=-1))

    final_votes = torch.cat(votes)
    mode_idx = torch.mode(final_votes)[0].item()
    return [mode_idx], max_queries


def main(args):
    # global EPS

    EPS = args.eps
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
    logging.info(f"当前运行设备: {device}")

    # 将数据集加载为 PyTorch Tensor 并推送至指定设备
    D = torch.tensor(D_np, dtype=torch.float32, device=device)
    n, d = D.shape
    logging.info(f"成功加载数据集: {n} 条数据, {d} 个维度。")

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]
    RS_N = args.N

    rs_results, at_results = [], []
    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))

    logging.info(f"开始执行 {num_experiments} 次对比实验...")

    for i in range(num_experiments):
        logging.info(f"\n--- 实验 {i + 1}/{num_experiments} ---")

        # 1. Oracle 生成 (在 GPU 上直接生成)
        true_u = dirichlet_dist.sample()
        true_e = E_min + torch.rand(d, device=device) * (E_max - E_min)

        distance = get_weighted_distance_sq(D, true_e, true_u)
        true_best_idx = torch.argmin(distance).item()
        best_distance = torch.min(distance).item()
        logging.info(f"  [Oracle] 用户的真实最优数据点索引为: {true_best_idx}")

        # 2. 运行 RS
        start_time = time.time()
        # 注意：此处返回的 rs_best_indices 是一个 List
        rs_best_indices, rs_queries = range_shrink_algorithm(D, true_e, true_u, RS_N, max_queries=200, eps=EPS)
        rs_time = time.time() - start_time

        # 评估指标计算
        if rs_best_indices and len(rs_best_indices) > 0:
            sorted_indices = torch.argsort(distance)

            # 分别计算列表中每一个点的 rank，并找出最差（最大）的 rank
            ranks = [(sorted_indices == idx).nonzero(as_tuple=True)[0][0].item() for idx in rs_best_indices]
            worst_rank = max(ranks)

            # 找到对应最差 rank 的那个索引
            worst_idx = rs_best_indices[ranks.index(worst_rank)]

            # 计算该最差点到真值的距离平方
            worst_distance_sq = distance[worst_idx].item()

            # Bias 计算 (保持原算法距离平方差的比例计算结构)
            dis_bias = (worst_distance_sq - best_distance) / (best_distance + 1e-12)

            # 命中判断：真实距离 d_i 是否在 d*(1+eps) 范围内。
            # 数学等价于：最差点的距离平方 <= 最佳点的距离平方 * (1 + eps)^2
            rs_success = worst_distance_sq <= best_distance * ((1.0 + EPS) ** 2)
        else:
            # 极端异常防崩溃兜底
            worst_rank = n - 1
            dis_bias = float('inf')
            rs_success = False

        rs_results.append({
            'queries': rs_queries,
            'success': rs_success,
            'time': rs_time,
            'dis_bias': dis_bias,
            'rank': worst_rank
        })

        logging.info(
            f"  [RS 算法] 返回交集大小: {len(rs_best_indices)} | 最差排名: {worst_rank + 1}/{n} | 最差偏差: {dis_bias:.4f} | 次数: {rs_queries} | 命中: {rs_success}")
    # 5. 统计报告
    logging.info("\n" + "=" * 40)
    logging.info(f"实验结果统计 ( {num_experiments}次平均)")
    logging.info("=" * 40)

    rs_avg_queries = np.mean([r['queries'] for r in rs_results if r['rank'] != (n-1)])
    rs_success_rate = np.mean([r['success'] for r in rs_results])
    rs_rank = np.mean([(r['rank']+1) for r in rs_results if r['rank'] != (n-1)])
    rs_top5 = np.sum([1 for r in rs_results if r['rank'] <= 5])
    rs_fail = np.sum([1 for r in rs_results if r['rank'] == (n-1)])
    rs_bias = np.mean([r['dis_bias'] for r in rs_results if r['rank'] != (n-1) ])
    rs_avg_time = np.mean([r['time'] for r in rs_results])


    logging.info(f"[Range Shrink] 采样点数: {RS_N}")
    logging.info(f"[Range Shrink] 命中率: {rs_success_rate:.4f}")
    logging.info(f"[Range Shrink] Top-5 命中率: {rs_top5:.4f}/{num_experiments}")
    logging.info(f"[Range Shrink] 平均排位: {rs_rank:.1f}/{n}")
    logging.info(f"[Range Shrink] 平均偏差: {rs_bias:.1f}")
    logging.info(f"[Range Shrink] 平均提问次数: {rs_avg_queries:.1f}")
    logging.info(f"[Range Shrink] 失败次数: {rs_fail}/{num_experiments}")
    logging.info(f"[Range Shrink] 平均单次查找总耗时: {rs_avg_time:.4f} 秒")


if __name__ == "__main__":
    main()