"""Shared numerical, monitoring and experiment utilities for AT and RS."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def _initialize_highs_before_torch():
    # Initialize HiGHS first to avoid the Windows numerical-runtime conflict.
    linprog(np.array([1.0]), A_ub=np.array([[-1.0]]), b_ub=np.array([0.0]),
            bounds=[(0, 1)], method="highs")


_initialize_highs_before_torch()

import torch


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"experiment_{timestamp}.log"

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
    logging.info("Log file created: %s", log_filename)

def get_weighted_distance_sq(D, e, u):
    """Return weighted squared distances with tensor broadcasting."""
    return torch.sum(u * (D - e) ** 2, dim=-1)


def simulate_user_oracle(p_idx, q_idx, D, true_e, true_u):
    """Simulate a strict preference; ties select the right-hand item."""
    dist_p = get_weighted_distance_sq(D[p_idx], true_e, true_u)
    dist_q = get_weighted_distance_sq(D[q_idx], true_e, true_u)
    return (dist_p < dist_q).item()

def load_dataset(filepath):
    """Load an n-by-d text dataset with an [n d] header and normalize columns."""
    try:
        with open(filepath, 'r') as f:
            raw_data = f.read().split()

        n, d = int(raw_data[0]), int(raw_data[1])

        D = np.array(raw_data[2:2 + n * d], dtype=float).reshape(n, d)
        col_min = D.min(axis=0)
        col_max = D.max(axis=0)

        # Keep constant columns finite when normalizing.
        range_val = np.maximum(col_max - col_min, 1e-8)

        D_normalized = (D - col_min) / range_val

        return D_normalized
    except Exception as e:
        print(f"Failed to load dataset; check the path and file format. Error: {e}")
        return None


def synchronize(device):
    """Complete GPU work before recording elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def initialize_monitor(data, D, true_e, true_u):
    """Reset a module's DATA in place, preserving the shared plotting schema."""
    best = int(torch.argmin(get_weighted_distance_sq(D, true_e, true_u)))
    data.clear()
    data.update({"Best P": D[best].tolist(), "E-U Route": [],
                 "True E-U": [true_e.tolist(), true_u.tolist()], "Time": []})
    data["RNG before anchor"] = torch.get_rng_state().tolist()
    if D.device.type == "cuda":
        data["CUDA RNG before anchor"] = torch.cuda.get_rng_state(D.device).tolist()


def record_query_state(data, e, u, started):
    """Record one post-feedback state and its matching elapsed-seconds timestamp."""
    synchronize(e.device)
    data["E-U Route"].append([e.tolist(), u.tolist()])
    data["Time"].append(time.perf_counter() - started)


def prepare_experiment(args, algorithm, dtype, output_dir=None):
    """Load a dataset and create an optional, non-overwriting JSON destination."""
    if args.rounds < 1 or args.data_size < 1:
        raise ValueError("rounds and data_size must be positive")
    values = load_dataset(args.data)
    if values is None:
        raise ValueError("Cannot load dataset; check the path and file format")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = torch.as_tensor(values[:args.data_size], dtype=dtype, device=device)
    n, d = D.shape
    logging.info("Dataset: %s | Items: %d | Dimensions: %d | Device: %s", args.data, n, d, device)
    output = None
    if getattr(args, "save_json", True):
        output = Path(output_dir or getattr(args, "output_dir", None) or
                      Path("experiments") / algorithm / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        output.mkdir(parents=True, exist_ok=False)
        (output / "routes").mkdir()
    return D, output


def evaluate_selection(distances, indices, eps, *, eps_power=1, bias_offset=1e-9):
    """Evaluate the worst returned item; preserve each algorithm's tolerance units."""
    best = int(torch.argmin(distances))
    best_distance = distances[best].item()
    if not indices:
        return dict(success=False, rank=len(distances) - 1, dis_bias=float("inf"), true_index=best)
    sorted_indices = torch.argsort(distances)
    ranks = [int((sorted_indices == p).nonzero(as_tuple=True)[0][0]) for p in indices]
    rank = max(ranks)
    distance = distances[indices[ranks.index(rank)]].item()
    return dict(success=distance <= best_distance * ((1 + eps) ** eps_power),
                rank=rank, dis_bias=(distance - best_distance) / (best_distance + bias_offset),
                true_index=best)


def save_trial(output, args, data, results, case, route_prefix):
    """Export the existing monitoring schema only when JSON output is enabled."""
    if output is not None:
        (output / "routes" / f"{route_prefix}_{case}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        (output / "results.json").write_text(
            json.dumps(dict(args=vars(args), results=results), indent=2), encoding="utf-8")


def run_experiments(args, algorithm, data, solve, dtype, *, output_dir=None,
                    route_prefix="eu_route", eps_power=1, bias_offset=1e-9):
    """Run shared trials; solve returns (item indices, question count, diagnostics)."""
    D, output = prepare_experiment(args, algorithm, dtype, output_dir)
    n, d = D.shape
    lower, upper = D.min(dim=0).values, D.max(dim=0).values
    prior = torch.distributions.Dirichlet(torch.ones(d, device=D.device))
    results = []
    for i in range(args.rounds):
        true_u = prior.sample()
        true_e = lower + torch.rand(d, device=D.device) * (upper - lower)
        distances = get_weighted_distance_sq(D, true_e, true_u)
        synchronize(D.device)
        started = time.perf_counter()
        indices, queries, diagnostics = solve(D, true_e, true_u, i)
        synchronize(D.device)
        elapsed = time.perf_counter() - started
        result = evaluate_selection(distances, indices, args.eps,
                                    eps_power=eps_power, bias_offset=bias_offset)
        result.update(queries=queries, time=elapsed, **diagnostics)
        assert queries == len(data["Time"]) == len(data["E-U Route"])
        assert queries == sum(event["cost"] for event in data["Comparisons"])
        data["Result"] = result
        results.append(result)
        logging.info("[%s %d/%d] Items: %s | Worst rank: %d/%d | Bias: %.4f | "
                     "Queries: %d | Success: %s | Stop: %s | Time: %.3fs",
                     algorithm, i + 1, args.rounds, indices, result["rank"] + 1, n,
                     result["dis_bias"], queries, result["success"], data["Termination"], elapsed)
        save_trial(output, args, data, results, i, route_prefix)
    logging.info("[%s Summary] Trials: %d | Success rate: %.4f | Mean queries: %.1f | Mean time: %.3fs",
                 algorithm, len(results), np.mean([r["success"] for r in results]),
                 np.mean([r["queries"] for r in results]), np.mean([r["time"] for r in results]))
    if output is not None:
        logging.info("JSON output saved to %s", output)
    return results
