"""Range shrink with particle filtering, hit-and-run replenishment and voting.

DATA uses the same plotting schema as AT. Every oracle question, including
tournament questions, adds one post-feedback state and elapsed timestamp.
"""
import time

# Import shared runtime initialization before importing PyTorch.
from utils import (get_weighted_distance_sq, initialize_monitor, record_query_state,
                   run_experiments, setup_logging, simulate_user_oracle)

import torch


DATA = {"Best P": [], "E-U Route": [], "True E-U": [], "Time": []}


def _particle_distances(D, e_samples, u_samples, chunk_size=5000):
    """Yield particle-to-item distances in bounded-size chunks."""
    for start in range(0, len(e_samples), chunk_size):
        e_chunk = e_samples[start:start + chunk_size].unsqueeze(1)
        u_chunk = u_samples[start:start + chunk_size].unsqueeze(1)
        yield get_weighted_distance_sq(D.unsqueeze(0), e_chunk, u_chunk)


def _nearest_votes(D, e_samples, u_samples):
    return torch.cat([torch.argmin(distances, dim=-1)
                      for distances in _particle_distances(D, e_samples, u_samples)])


def _replenish_particles(D, e_samples, u_samples, valid_mask, H, E_min, E_max):
    """Replace rejected particles with feasible hit-and-run proposals."""
    device = e_samples.device
    survivors_e, survivors_u = e_samples[valid_mask], u_samples[valid_mask]
    num_survivors = len(survivors_e)
    num_dead = len(e_samples) - num_survivors
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


def range_shrink_algorithm(D, true_e, true_u, initial_N=5000, max_queries=200, eps=0.0):
    """Return candidate indices and the actual number of oracle questions.

    max_queries limits filtering rounds. As before, a final tournament may ask
    additional questions. The route represents the surviving-particle mean;
    tournament answers update the incumbent but do not fit new particles, so
    those entries retain the last available particle mean.
    """
    if initial_N < 1 or max_queries < 0:
        raise ValueError("initial_N must be positive and max_queries must be nonnegative")
    if D.ndim != 2 or not len(D) or not D.shape[1]:
        raise ValueError("D must be a nonempty two-dimensional dataset")
    started = time.perf_counter()
    initialize_monitor(DATA, D, true_e, true_u)
    DATA["Comparisons"], DATA["Feedback"] = [], []
    device = D.device
    n, d = D.shape
    H = []
    E_min, E_max = D.min(dim=0).values, D.max(dim=0).values
    e_samples = E_min + torch.rand(initial_N, d, dtype=D.dtype, device=device) * (E_max - E_min)
    prior = torch.distributions.Dirichlet(torch.ones(d, dtype=D.dtype, device=device))
    u_samples = prior.sample((initial_N,))
    DATA["Initial E-U"] = [e_samples.mean(dim=0).tolist(), u_samples.mean(dim=0).tolist()]
    queries = 0

    def ask(left, right, source):
        nonlocal queries
        prefers_left = simulate_user_oracle(left, right, D, true_e, true_u)
        winner, loser = (left, right) if prefers_left else (right, left)
        queries += 1
        DATA["Comparisons"].append(dict(left=left, right=right, source=source,
                                        cached=False, cost=1, prefers_left=prefers_left))
        DATA["Feedback"].append((winner, loser))
        return winner, loser

    def finish(indices, reason):
        DATA["Termination"] = reason
        return indices, queries

    def tournament(e_particles, u_particles, reason):
        candidates = torch.unique(_nearest_votes(D, e_particles, u_particles))
        best = int(candidates[0])
        mean_e, mean_u = e_particles.mean(dim=0), u_particles.mean(dim=0)
        for competitor in candidates[1:].tolist():
            best, _ = ask(best, competitor, "tournament")
            record_query_state(DATA, mean_e, mean_u, started)
        return finish([best], reason)

    if n == 1:
        return finish([0], "singleton")

    for round_index in range(max_queries):
        # Apply the new answer and filter against all accepted constraints.
        perm = torch.randperm(n, device=device)
        winner, loser = ask(int(perm[0]), int(perm[1]), "filter")
        H.append((winner, loser))
        valid_mask = torch.ones(initial_N, dtype=torch.bool, device=device)
        for w_idx, l_idx in H:
            dist_w = get_weighted_distance_sq(D[w_idx], e_samples, u_samples)
            dist_l = get_weighted_distance_sq(D[l_idx], e_samples, u_samples)
            valid_mask &= dist_w < dist_l

        if not valid_mask.any():
            # Keep the last available cloud when the new answer rejects it all.
            # Record this answer as well as every later tournament answer.
            record_query_state(DATA, e_samples.mean(dim=0), u_samples.mean(dim=0), started)
            return tournament(e_samples, u_samples, "empty_cloud_tournament")

        survivors_e, survivors_u = e_samples[valid_mask], u_samples[valid_mask]
        mean_e, mean_u = survivors_e.mean(dim=0), survivors_u.mean(dim=0)
        intersection = torch.ones(n, dtype=torch.bool, device=device)
        for distances in _particle_distances(D, survivors_e, survivors_u):
            # RS eps measures distance, so squared distances use (1 + eps)^2.
            threshold = distances.min(dim=-1, keepdim=True).values * ((1 + eps) ** 2)
            intersection &= torch.all(distances <= threshold, dim=0)
        indices = torch.nonzero(intersection).squeeze(-1).tolist()
        if indices:
            record_query_state(DATA, mean_e, mean_u, started)
            return finish(indices, "intersection")

        if round_index + 1 == max_queries:
            record_query_state(DATA, mean_e, mean_u, started)
            return tournament(survivors_e, survivors_u, "max_queries_tournament")

        _replenish_particles(D, e_samples, u_samples, valid_mask, H, E_min, E_max)
        record_query_state(DATA, mean_e, mean_u, started)

    # With a zero filtering budget, return the initial cloud's modal vote.
    votes = _nearest_votes(D, e_samples, u_samples)
    return finish([int(torch.mode(votes).values)], "max_queries")


def main(args, output_dir=None):
    """Run RS trials with shared monitoring, evaluation and optional JSON export."""
    if args.N < 1:
        raise ValueError("N must be positive")
    setup_logging()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    def solve(D, true_e, true_u, case):
        indices, queries = range_shrink_algorithm(D, true_e, true_u, args.N, eps=args.eps)
        return indices, queries, dict(returned_indices=indices, termination=DATA["Termination"],
                                      comparison_requests=len(DATA["Comparisons"]), cache_hits=0)

    return run_experiments(args, "RS", DATA, solve, dtype, output_dir=output_dir,
                           route_prefix="rs_route", eps_power=2, bias_offset=1e-12)


if __name__ == "__main__":
    from run import parse_args
    main(parse_args().parse_args())
