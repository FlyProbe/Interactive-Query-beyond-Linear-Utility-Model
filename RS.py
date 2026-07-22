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


# ==========================================
# 0. 日志配置函数
# ==========================================
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"experiment_{timestamp}.log")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            stream_handler
        ],
        force=True
    )
    logging.info(f"日志文件已创建: {log_filename}")


# ==========================================
# 1. 通用辅助函数 (已重构为 PyTorch 算子)
# ==========================================
def get_weighted_distance_sq(D, e, u):
    """计算数据集 D 中所有点到期望点 e 的加权距离平方 (支持多维广播)"""
    return torch.sum(u * (D - e) ** 2, dim=-1)


def simulate_user_oracle(p_idx, q_idx, D, true_e, true_u):
    """模拟用户的真实选择 (Oracle)"""
    dist_p = get_weighted_distance_sq(D[p_idx], true_e, true_u)
    dist_q = get_weighted_distance_sq(D[q_idx], true_e, true_u)
    return (dist_p < dist_q).item()


# ==========================================
# 2. RS 算法 (Range Shrink - 基于 SMC 粒子重采样优化)
# ==========================================
def range_shrink_algorithm(D, true_e, true_u, initial_N=5000, max_queries = 200):
    """
    SMC + Hit-and-Run (贴边游走) 架构
    """
    device = D.device
    n, d = D.shape
    H = []  # 记录偏好约束，格式为 (winner_idx, loser_idx)

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]

    # 全局初始化 N 个粒子 (e, u)
    e_samples = E_min + torch.rand(initial_N, d, device=device) * (E_max - E_min)
    dirichlet_dist = torch.distributions.Dirichlet(torch.ones(d, device=device))
    u_samples = dirichlet_dist.sample((initial_N,))

    rounds = 0
    from tqdm import tqdm
    tbar = tqdm(total=max_queries, desc="Hit-and-Run Interacting")

    while rounds < max_queries:
        rounds += 1
        tbar.update(1)

        # ---------------------------------------------------------
        # 1. 产生交互与获取反馈
        # ---------------------------------------------------------
        perm = torch.randperm(n, device=device)
        p_idx, q_idx = perm[0].item(), perm[1].item()

        if simulate_user_oracle(p_idx, q_idx, D, true_e, true_u):
            winner_idx, loser_idx = p_idx, q_idx
        else:
            winner_idx, loser_idx = q_idx, p_idx

        H.append((winner_idx, loser_idx))

        # ---------------------------------------------------------
        # 2. 筛选幸存者
        # ---------------------------------------------------------
        valid_mask = torch.ones(initial_N, dtype=torch.bool, device=device)
        for w_idx, l_idx in H:
            dist_w = get_weighted_distance_sq(D[w_idx], e_samples, u_samples)
            dist_l = get_weighted_distance_sq(D[l_idx], e_samples, u_samples)
            valid_mask &= (dist_w < dist_l)

        num_survivors = valid_mask.sum().item()
        num_dead = initial_N - num_survivors

        # ---------------------------------------------------------
        # 3. 锦标赛兜底 (如果当前反馈导致全军覆没)
        # ---------------------------------------------------------
        if num_survivors == 0:
            H.pop()  # 撤销致命约束
            # 利用上一轮的合法粒子重新计算投票
            prev_votes = []
            chunk_size = 5000
            for i in range(0, initial_N, chunk_size):
                e_chunk = e_samples[i: i + chunk_size].unsqueeze(1)
                u_chunk = u_samples[i: i + chunk_size].unsqueeze(1)
                dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
                prev_votes.append(torch.argmin(dist_chunk, dim=-1))

            unique_candidates = torch.unique(torch.cat(prev_votes))

            # 对幸存的候选答案进行锦标赛两两对决
            best_idx = unique_candidates[0].item()
            for i in range(1, len(unique_candidates)):
                competitor_idx = unique_candidates[i].item()
                rounds += 1
                tbar.update(1)
                if not simulate_user_oracle(best_idx, competitor_idx, D, true_e, true_u):
                    best_idx = competitor_idx
            tbar.close()
            return best_idx, rounds

        # ---------------------------------------------------------
        # 4. 存活者投票收敛判定
        # ---------------------------------------------------------
        survivors_e = e_samples[valid_mask]
        survivors_u = u_samples[valid_mask]

        votes = []
        chunk_size = 5000
        for i in range(0, num_survivors, chunk_size):
            e_chunk = survivors_e[i: i + chunk_size].unsqueeze(1)
            u_chunk = survivors_u[i: i + chunk_size].unsqueeze(1)
            dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
            votes.append(torch.argmin(dist_chunk, dim=-1))

        unique_votes = torch.unique(torch.cat(votes))

        if len(unique_votes) == 1:
            tbar.close()
            return unique_votes[0].item(), rounds - 1
        elif len(unique_votes) <= 5 and rounds == max_queries:
            best_idx = unique_votes[0].item()
            for i in range(1, len(unique_votes)):
                competitor_idx = unique_votes[i].item()
                rounds += 1
                tbar.update(1)
                if not simulate_user_oracle(best_idx, competitor_idx, D, true_e, true_u):
                    best_idx = competitor_idx
            tbar.close()
            return best_idx, rounds

        # ---------------------------------------------------------
        # 5. Hit-and-Run 定向贴边补齐阵亡粒子
        # ---------------------------------------------------------
        if num_dead > 0:
            clone_idx = torch.randint(0, num_survivors, (num_dead,), device=device)
            c_e = survivors_e[clone_idx]
            c_u = survivors_u[clone_idx]

            # 生成随机方向 (归一化以保证计算稳定性)
            Delta_e = torch.randn_like(c_e)
            Delta_e = Delta_e / torch.norm(Delta_e, dim=-1, keepdim=True)

            Delta_u = torch.randn_like(c_u)
            Delta_u = Delta_u - Delta_u.mean(dim=-1, keepdim=True)  # 保证 sum = 0
            Delta_u = Delta_u / (torch.norm(Delta_u, dim=-1, keepdim=True) + 1e-9)

            # --- A. 计算物理边界 t_phys ---
            t_e_min = torch.where(Delta_e < 0, (E_min - c_e) / Delta_e, float('inf'))
            t_e_max = torch.where(Delta_e > 0, (E_max - c_e) / Delta_e, float('inf'))
            t_e = torch.minimum(t_e_min.min(dim=-1)[0], t_e_max.min(dim=-1)[0])

            t_u_min = torch.where(Delta_u < 0, (0.001 - c_u) / Delta_u, float('inf'))
            t_u_max = torch.where(Delta_u > 0, (1.0 - c_u) / Delta_u, float('inf'))
            t_u = torch.minimum(t_u_min.min(dim=-1)[0], t_u_max.min(dim=-1)[0])

            t_phys = torch.minimum(t_e, t_u)

            # --- B. 计算逻辑边界 t_logic (二次项完美抵消的一元二次方程) ---
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

                # 求一元二次方程根 (掩码处理无解或负解的情况)
                # t1 = (-b + sqrt(disc)) / 2a ; t2 = (-b - sqrt(disc)) / 2a
                t1 = torch.where(a != 0, (-b + torch.sqrt(torch.clamp(disc, min=0))) / (2 * a + 1e-12),
                                 -c / (b + 1e-12))
                t2 = torch.where(a != 0, (-b - torch.sqrt(torch.clamp(disc, min=0))) / (2 * a + 1e-12), float('inf'))

                # 寻找最小正实根 (排除漂移的负数或极小值)
                t1 = torch.where((disc >= 0) & (t1 > 1e-6), t1, float('inf'))
                t2 = torch.where((disc >= 0) & (t2 > 1e-6), t2, float('inf'))

                t_h_min = torch.minimum(t1, t2)
                t_logic = torch.minimum(t_logic, t_h_min)

            # --- C. 贴边跃迁  ---
            t_max = torch.minimum(t_phys, t_logic)

            # 0.9比例
            # t_step = 0.9 * t_max

            # 随机
            rand_ratios = torch.rand(num_dead, device=device)
            t_step = rand_ratios * 0.999 * t_max

            new_e = c_e + t_step.unsqueeze(-1) * Delta_e
            new_u = c_u + t_step.unsqueeze(-1) * Delta_u

            # 安全归一化兜底 (防止浮点精度误差导致越界)
            new_e = torch.max(torch.min(new_e, E_max), E_min)
            new_u = torch.clamp(new_u, 0.001, 1.0)
            new_u = new_u / new_u.sum(dim=-1, keepdim=True)

            e_samples[~valid_mask] = new_e
            u_samples[~valid_mask] = new_u

    tbar.close()

    # 达到 max_queries 未收敛，取最后状态投票的众数作为 Fallback
    votes = []
    for i in range(0, initial_N, 5000):
        e_chunk = e_samples[i: i + 5000].unsqueeze(1)
        u_chunk = u_samples[i: i + 5000].unsqueeze(1)
        dist_chunk = torch.sum(u_chunk * (D.unsqueeze(0) - e_chunk) ** 2, dim=-1)
        votes.append(torch.argmin(dist_chunk, dim=-1))

    final_votes = torch.cat(votes)
    mode_idx = torch.mode(final_votes)[0].item()
    return mode_idx, max_queries


# ==========================================
# 4. 数据加载与 Main 实验组装
# ==========================================
def load_dataset(filepath):
    """解析带有 [n d] 开头的文本数据集"""
    try:
        with open(filepath, 'r') as f:
            raw_data = f.read().split()

        n, d = int(raw_data[0]), int(raw_data[1])

        D = np.array(raw_data[2:2 + n * d], dtype=float).reshape(n, d)
        col_min = D.min(axis=0)
        col_max = D.max(axis=0)

        # 计算极差，并加入 1e-8 的微小偏移量防止除以 0 的灾难（当某一整列数值全相同时）
        range_val = np.maximum(col_max - col_min, 1e-8)

        # 归一化公式： (X - X_min) / (X_max - X_min)
        D_normalized = (D - col_min) / range_val

        return D_normalized
    except Exception as e:
        print(f"数据加载失败，请检查文件路径和格式。错误: {e}")
        return None

def main():
    setup_logging()  # 初始化日志

    filepath = 'datasets/nba.txt'
    D_np = load_dataset(filepath)
    if D_np is None:
        return
    num_experiments = 10

    # 自动识别并配置 GPU / CPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"当前运行设备: {device}")

    # 将数据集加载为 PyTorch Tensor 并推送至指定设备
    D = torch.tensor(D_np, dtype=torch.float32, device=device)
    n, d = D.shape
    logging.info(f"成功加载数据集: {n} 条数据, {d} 个维度。")

    E_min = torch.min(D, dim=0)[0]
    E_max = torch.max(D, dim=0)[0]
    # num_experiments = 14
    RS_N = 5000

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
        rs_best_idx, rs_queries = range_shrink_algorithm(D, true_e, true_u, RS_N)
        rs_time = time.time() - start_time
        rs_success = (rs_best_idx == true_best_idx)

        # 评估指标计算 (处理收敛失败返回 -1 的边界情况，防止原本的 distance 报错问题)
        if rs_best_idx != -1:
            sorted_indices = torch.argsort(distance)
            rank = (sorted_indices == rs_best_idx).nonzero(as_tuple=True)[0][0].item()
            dis_bias = (get_weighted_distance_sq(D[rs_best_idx], true_e, true_u).item() - best_distance) / best_distance
        else:
            rank = n - 1
            dis_bias = float('inf')

        rs_results.append(
            {'queries': rs_queries, 'success': rs_success, 'time': rs_time, "dis_bias": dis_bias, "rank": rank})

        logging.info(
            f"  [RS 算法] 返回点: {rs_best_idx} | 排名: {rank + 1}/{n} | 偏差: {dis_bias:.4f} | 次数: {rs_queries} | 命中: {rs_success}")

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