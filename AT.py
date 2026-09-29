"""Anchor tracking with candidate information gain (the selected ACIG strategy).

Select one unseen question, update the anchor, then replan both chains.
Information gain scores questions only; the stopping decision uses the LP.
DATA keeps the existing monitoring schema. Each Time / E-U Route entry is
recorded after one paid question and its anchor update; cached answers add none.
"""
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


# Import shared runtime initialization before importing PyTorch.
from utils import (get_weighted_distance_sq, initialize_monitor, record_query_state,
                   run_experiments, setup_logging)

import torch
import torch.nn.functional as F


DATA = {"Best P": [], "E-U Route": [], "True E-U": [], "Time": []}


def configure(dtype="float32"):
    """Configure the tensor dtype and matching numerical tolerances."""
    global TORCH_DTYPE, LP_ACC, LP_SAFE, ETA
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    TORCH_DTYPE = torch.float64 if dtype == "float64" else torch.float32
    LP_ACC, LP_SAFE, ETA = (
        (1e-8, 1e-9, 1e-10) if dtype == "float64" else (1e-6, 1e-7, 1e-7))


configure()


class ComparisonCache:
    """Reuse exact deterministic answers and count only previously unseen pairs."""

    def __init__(self, D, oracle_e, oracle_u):
        self.D, self.e, self.u = D, oracle_e, oracle_u
        self.answers = {}
        self.events = []
        self.queries_used = 0

    def compare(self, left, right, source):
        key = (min(left, right), max(left, right))
        cached = key in self.answers
        if left == right:
            outcome, cost = 0, 0
        elif cached:
            outcome, cost = self.answers[key], 0
        else:
            a = get_weighted_distance_sq(self.D[key[0]], self.e, self.u)
            b = get_weighted_distance_sq(self.D[key[1]], self.e, self.u)
            outcome = -1 if a < b else (1 if b < a else 0)
            self.answers[key] = outcome
            self.queries_used += 1
            cost = 1
        prefers_left = outcome == (-1 if left == key[0] else 1)
        self.events.append(dict(left=left, right=right, source=source,
                                cached=cached, cost=cost, prefers_left=prefers_left))
        return prefers_left, cost


def at_find_nearest_neighbor(D: torch.Tensor, e: torch.Tensor, u: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Return the nearest item under the current weighted squared distance."""
    distances = get_weighted_distance_sq(D, e, u)

    p_idx = torch.argmin(distances).item()

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
    """Return the first feasible challenger and its anchor, or certify LP termination."""
    _, d = D.shape
    p_star = D[p_idx]

    # Lift feedback to linear constraints in (u, z), where z = u * e.
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

    D_np = D.cpu().numpy()
    p_star_np = p_star.cpu().numpy()

    u_coefs_obj = D_np ** 2 - p_star_np ** 2
    z_coefs_obj = -2 * (D_np - p_star_np)
    C_all = np.concatenate([u_coefs_obj, z_coefs_obj], axis=-1)

    current_distances = get_weighted_distance_sq(D, e, u)
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

            return q_idx, e_prime_tensor, u_prime_tensor

    return None, None, None


def at_construct_q_chain_alternating(D: torch.Tensor, e: torch.Tensor, u: torch.Tensor,
                                     e_prime: torch.Tensor, u_prime: torch.Tensor,
                                     vary_u: bool):
    """Build a lower-envelope chain by varying weights or the ideal point."""
    if vary_u:
        # Fix the rebuttal ideal point and vary the weights.
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
        # Fix the rebuttal weights and vary the ideal point.
        delta_e = e_prime - e
        # A_p = \sum u_prime_i (p_i - e_i)^2
        A = torch.sum(u_prime * (D - e) ** 2, dim=-1)
        # Point-independent terms cancel when comparing intersections.
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


def at_alternating_learning(
        D: torch.Tensor,
        history_constraints: List[Tuple[int, int]],
        current_e: torch.Tensor,
        current_u: torch.Tensor,
        max_iter: int = 500,
        init_lr: float = 0.5,
        device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alternate feasible likelihood descent in weights and the ideal point."""
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
        """Return the negative log-likelihood and strict feedback feasibility."""
        dist_P = torch.sum(u_test * (P - e_test) ** 2, dim=-1)
        dist_Q = torch.sum(u_test * (Q - e_test) ** 2, dim=-1)
        delta = dist_P - dist_Q

        L = torch.sum(F.softplus(delta)).item()

        is_valid = torch.all(delta < 0).item()

        return L, is_valid

    for i in range(max_iter):
        e_prev, u_prev = e.clone(), u.clone()

        # Update weights using backtracking while preserving all feedback.
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

            if L_next < L_prev and is_valid:
                u = u_next
                break

            lr *= 0.5

        # Update the ideal point with the accepted weights held fixed.
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

        if i > 50 and torch.linalg.norm(e - e_prev) < 1e-5 and torch.linalg.norm(u - u_prev) < 1e-5:
            break
    return e, u


def row_dots(rows, vector):
    # Small 2d-wide rows: avoid NumPy BLAS starting a second OpenMP runtime
    # alongside PyTorch on the existing Anaconda/Windows installation.
    return np.sum(rows*vector, axis=1)


def sample_gaps(samples, rows):
    return np.einsum("si,hi->sh", samples, rows, optimize=False)


@dataclass(frozen=True)
class SelectionConfig:
    samples: int = 256
    burn_in: int = 64
    thinning: int = 2
    seed: int = 20260924
    tolerance: float = 1e-9

    def __post_init__(self):
        if self.samples < 2 or self.burn_in < 0 or self.thinning < 1 or self.tolerance <= 0:
            raise ValueError("Invalid sampling configuration")


def candidate_pairs(u_chain, e_chain, incumbent, rebuttal, answered):
    """Union of unseen adjacent pairs from both alternating chains and the rebuttal pair."""
    candidates = {}
    for source, pairs in (
        ("rebuttal", [(incumbent, rebuttal)]),
        ("u_chain", zip(u_chain, u_chain[1:])),
        ("e_chain", zip(e_chain, e_chain[1:])),
    ):
        for left, right in pairs:
            key = tuple(sorted((int(left), int(right))))
            if key[0] == key[1] or key in answered:
                continue
            candidates.setdefault(key, []).append(source)
    return candidates


def balanced_pair(points, samples, candidates, tolerance=1e-9):
    """Maximize empirical answer disagreement, NOT best-item information gain.

    Near-ties abstain, so a degenerate all-tie cloud cannot appear informative.
    If the cloud is unavailable/uninformative, prefer the unseen rebuttal pair.
    """
    if not candidates:
        return None, {"reason": "no_unasked_pair"}
    ranked = []
    for pair, sources in candidates.items():
        left, right = points[list(pair)]
        row = np.r_[left*left-right*right, -2*(left-right)]
        norm = np.sqrt(np.sum(row*row))
        gaps = row_dots(samples,row/max(norm, 1e-15)) if len(samples) else np.empty(0)
        negative, positive = int((gaps < -tolerance).sum()), int((gaps > tolerance).sum())
        score = min(negative, positive)/len(samples) if len(samples) else 0.
        ties = len(samples)-negative-positive
        probability = (negative+.5*ties)/len(samples) if len(samples) else None
        ranked.append((score, int("rebuttal" in sources), pair,
                       dict(score=score, left_probability=probability, sources=sources,
                            samples=len(samples), abstentions=ties)))
    # Deterministic tie break; no oracle or anchor-distance lookahead.
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    score, _, pair, info = ranked[0]
    info["reason"] = "balanced_split" if score > 0 else "uninformative_fallback"
    return pair, info


class FeasibleSamples:
    """Hit-and-run proxy ensemble in the lifted feedback polytope.

    Sampling in (u,z) is used only for query scoring.
    It approximates uniform lifted-space volume, not a posterior for the
    original uniform-e/Dirichlet-u prior. Finite-chain mixing is not certified.
    """
    def __init__(self, points, config=SelectionConfig()):
        self.points = np.asarray(points, dtype=np.float64)
        self.config = config
        self.rng = np.random.default_rng(config.seed)  # independent of torch/oracle RNG
        self.samples = np.empty((0, 2*self.points.shape[1]))
        self.last_status = "not_sampled"

    def inequalities(self, history):
        d = self.points.shape[1]
        I, Z = np.eye(d), np.zeros((d,d))
        A = np.vstack((np.c_[-I,Z], np.c_[Z,-I], np.c_[-I,I]))
        if history:
            winners, losers = np.asarray(history, dtype=int).T
            p, q = self.points[winners], self.points[losers]
            feedback = np.c_[p*p-q*q, -2*(p-q)]
            norms = np.linalg.norm(feedback, axis=1)
            A = np.vstack((A, feedback[norms>0]/norms[norms>0,None]))
        return A

    def draw(self, history):
        d, config = self.points.shape[1], self.config
        A = self.inequalities(history)
        if len(self.samples):
            survivors = self.samples[np.all(sample_gaps(self.samples,A) <= config.tolerance, axis=1)]
        else:
            survivors = self.samples
        if len(survivors):
            state = survivors.mean(axis=0)
        else:
            # Interior starting point for the sampler only; never returned as an anchor.
            projected = A.copy()
            projected[:, :d] -= projected[:, :d].mean(axis=1, keepdims=True)
            radii = np.linalg.norm(projected, axis=1)
            eq = np.zeros((1,2*d+1)); eq[0,:d] = 1.
            result = linprog(np.r_[np.zeros(2*d),-1.],
                             A_ub=np.c_[A,radii], b_ub=np.zeros(len(A)),
                             A_eq=eq, b_eq=[1.], bounds=[(None,None)]*(2*d)+[(0,1)],
                             method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                                      "dual_feasibility_tolerance": 1e-9})
            if not result.success:
                self.last_status = "sampler_lp_failed"
                self.samples = np.empty((0,2*d))
                return self.samples
            state = result.x[:2*d]
        draws = []
        for step in range(config.burn_in+config.samples*config.thinning):
            direction = self.rng.normal(size=2*d)
            direction[:d] -= direction[:d].mean()
            direction /= np.sqrt(np.sum(direction*direction))
            slack, slopes = -row_dots(A,state), row_dots(A,direction)
            if np.min(slack) < -config.tolerance:
                break
            slack = np.maximum(slack, 0.)
            positive, negative = slopes > 1e-14, slopes < -1e-14
            low = np.max(slack[negative]/slopes[negative], initial=-np.inf)
            high = np.min(slack[positive]/slopes[positive], initial=np.inf)
            if not np.isfinite(low+high) or low > high:
                break
            state = state + (low+(high-low)*self.rng.uniform(1e-7,1-1e-7))*direction
            if step >= config.burn_in and (step-config.burn_in)%config.thinning == 0:
                if np.all(row_dots(A,state) <= config.tolerance):
                    draws.append(state.copy())
        self.samples = np.asarray(draws).reshape(-1,2*d)
        self.last_status = "ok" if len(draws)==config.samples else "incomplete_cloud"
        return self.samples


def best_item_labels(points, samples, point_chunk=512):
    """Exact argmin over ALL dataset points for each scoring sample.

    Scores omit the point-independent sum(u*e^2). Chunking bounds memory;
    non-BLAS einsum avoids the Windows NumPy/PyTorch OpenMP conflict.
    """
    points = np.asarray(points,dtype=np.float64)
    samples = np.asarray(samples,dtype=np.float64)
    if points.ndim != 2 or not len(points) or point_chunk < 1:
        raise ValueError("Nonempty 2D points and a positive chunk size required")
    d = points.shape[1]
    if samples.ndim != 2 or samples.shape[1] != 2*d:
        raise ValueError("Expected samples in (u,z) coordinates")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(samples)):
        raise ValueError("Points and samples must be finite")
    if not len(samples):
        return np.empty(0,dtype=int)
    best_scores = np.full(len(samples),np.inf)
    labels = np.full(len(samples),-1,dtype=int)
    for start in range(0,len(points),point_chunk):
        chunk = points[start:start+point_chunk]
        scores = (np.einsum("si,pi->sp",samples[:,:d],chunk*chunk,optimize=False)
                  -2*np.einsum("si,pi->sp",samples[:,d:],chunk,optimize=False))
        local = np.argmin(scores,axis=1)
        values = scores[np.arange(len(samples)),local]
        improve = values < best_scores  # exact ties retain the lowest dataset index
        labels[improve] = start+local[improve]
        best_scores[improve] = values[improve]
    return labels


def entropy(counts):
    counts = np.asarray(counts,dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.
    probabilities = counts[counts>0]/total
    return float(-np.sum(probabilities*np.log2(probabilities)))


def label_information_gain(labels, left_probabilities):
    """Empirical I(best-item label; binary answer), in bits.

    Near-tie parameter samples can give each answer half their mass, rather
    than inventing an informative third response or discarding their labels.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(left_probabilities,dtype=np.float64)
    if labels.ndim!=1 or probabilities.shape!=labels.shape:
        raise ValueError("Labels and probabilities must be equally sized vectors")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError("Answer probabilities must lie in [0,1]")
    if not len(labels):
        return dict(information_gain=0.,prior_entropy=0.,expected_entropy=0.,left_probability=None)
    _,inverse = np.unique(labels,return_inverse=True)
    counts = np.bincount(inverse)
    left_counts = np.bincount(inverse,weights=probabilities,minlength=len(counts))
    right_counts = counts-left_counts
    p_left = float(probabilities.mean())
    prior = entropy(counts)
    expected = p_left*entropy(left_counts)+(1-p_left)*entropy(right_counts)
    return dict(information_gain=max(0.,prior-expected),prior_entropy=prior,
                expected_entropy=expected,left_probability=p_left)


def select_pair(points, samples, candidates, tolerance=1e-9):
    """Maximize best-item information gain, with a rebuttal/index tie break.

    Labels are exact minimizers over ALL dataset items for each sampled
    parameter. Probabilities are empirical proxy probabilities, not a certified
    posterior. Near ties receive half mass, using a binary answer model.
    Empty samples or globally zero gain fall back to balanced-answer selection.
    """
    if not candidates:
        return None, dict(reason="no_unasked_pair", fallback=False)
    labels = best_item_labels(points, samples) if len(samples) else np.empty(0, dtype=int)
    diagnostics = dict(samples=len(samples), distinct_best_items=len(np.unique(labels)),
                       candidate_count_scored=len(candidates))
    ranked = []
    for pair, sources in candidates.items():
        left, right = points[list(pair)]
        row = np.r_[left*left-right*right, -2*(left-right)]
        norm = np.sqrt(np.sum(row*row))
        gaps = row_dots(samples, row/max(norm, 1e-15))
        probabilities = np.where(gaps < -tolerance, 1., np.where(gaps > tolerance, 0., .5))
        info = label_information_gain(labels, probabilities)
        info.update(sources=sources, near_ties=int(np.count_nonzero(probabilities == .5)))
        ranked.append((pair, info))
    best_gain = max(info["information_gain"] for _, info in ranked)
    if not len(samples) or best_gain <= 1e-12:
        pair, original = balanced_pair(points, samples, candidates, tolerance)
        info = next(info for key, info in ranked if key == pair)
        return pair, dict(original, **diagnostics, **info, fallback=True,
                          fallback_reason="empty_cloud" if not len(samples) else "zero_gain",
                          balanced_reason=original["reason"],
                          reason="balanced_fallback")
    tied = [(pair, info) for pair, info in ranked if best_gain-info["information_gain"] <= 1e-12]
    pair, info = min(tied, key=lambda entry: (-int("rebuttal" in entry[1]["sources"]), entry[0]))
    return pair, dict(**diagnostics, **info, fallback=False, reason="best_item_information_gain")


def anchor_tracking_algorithm(D, true_e, true_u, max_round=500,
                              config=SelectionConfig(), initial_anchor=None):
    """Each question is followed by an anchor update and full replanning.

    initial_anchor enables exact pairing without sharing the query-sampler RNG.
    max_round retains the 500-round safeguard; zero-query relocations count too.
    """
    started = time.perf_counter()
    data = DATA
    initialize_monitor(data, D, true_e, true_u)
    if initial_anchor is None:
        lower, upper = D.min(dim=0).values, D.max(dim=0).values
        e = lower+torch.rand(D.shape[1], device=D.device)*(upper-lower)
        u = torch.distributions.Dirichlet(torch.ones(D.shape[1],device=D.device)).sample()
    else:
        e, u = [torch.as_tensor(x,dtype=D.dtype,device=D.device).clone() for x in initial_anchor]
    data["Initial E-U"] = [e.tolist(), u.tolist()]
    cache = ComparisonCache(D,true_e,true_u)
    data["Comparisons"], data["C Decisions"] = cache.events, []
    history = []
    data["Feedback"] = history
    points = D.detach().cpu().numpy().astype(np.float64)
    sampler = FeasibleSamples(points,config)
    rounds = 0
    while rounds < max_round:
        p, _ = at_find_nearest_neighbor(D,e,u)
        # A cache may already prove certain challengers lose to this incumbent.
        # These exclusions are rebuilt for each incumbent, never carried globally.
        known_losers = set()
        for (left,right), outcome in cache.answers.items():
            if outcome == -1 and p == left:
                known_losers.add(right)
            elif outcome == 1 and p == right:
                known_losers.add(left)
        q, ep, up = at_check_stopping_condition_LP(
            history,D,p,e,u,known_losers,device=D.device)
        if q is None:
            data["Termination"] = "LP"
            data["Feedback"] = history
            return p,rounds,cache.queries_used

        key = tuple(sorted((p,q)))
        if key in cache.answers:
            # Relocate the anchor when a known answer already rejects the incumbent.
            prefers_p, _ = cache.compare(p,q,"cached_rebuttal")
            if not prefers_p and cache.answers[key] != 0:
                e,u = ep,up
                data["C Decisions"].append(dict(round=rounds,action="cached_relocation",pair=[p,q]))
                rounds += 1
                continue
            # The strict LP and an exact tied answer can disagree; do not loop or
            # certify convergence based on sample consensus / a missing question.
            data["Termination"] = "cached_rebuttal_conflict"
            data["Feedback"] = history
            return p,rounds,cache.queries_used

        u_chain, _ = at_construct_q_chain_alternating(D,e,u,ep,up,True)
        e_chain, _ = at_construct_q_chain_alternating(D,e,u,ep,up,False)
        candidates = candidate_pairs(u_chain,e_chain,p,q,cache.answers)
        samples = sampler.draw(history)
        pair, info = select_pair(points,samples,candidates,config.tolerance)
        if pair is None:
            raise RuntimeError("AT has an unseen rebuttal but no candidate question")
        left,right = pair
        source = "C:"+"+".join(info["sources"])
        prefers_left,cost = cache.compare(left,right,source)
        if cost != 1:
            raise RuntimeError("AT selected an already answered question")
        winner,loser = (left,right) if prefers_left else (right,left)
        history.append((winner,loser))
        if pair == key and winner == q:
            e,u = ep,up  # use the feasible rebuttal anchor
            update = "A_rebuttal_jump"
        else:
            e,u = at_alternating_learning(D,history,e,u,device=D.device)
            update = "A_alternating_learning"
        record_query_state(data, e, u, started)
        data["C Decisions"].append(dict(round=rounds,action="ask_one",pair=list(pair),
                                         winner=winner,incumbent=p,rebuttal=q,
                                         u_chain_length=len(u_chain),e_chain_length=len(e_chain),
                                         candidate_count=len(candidates),sampler_status=sampler.last_status,
                                         anchor_update=update,**info))
        rounds += 1
    data["Termination"] = "max_round"
    data["Feedback"] = history
    p,_ = at_find_nearest_neighbor(D,e,u)
    return p,rounds,cache.queries_used


def main(args, output_dir=None, config=SelectionConfig()):
    """Run independent AT trials with shared monitoring and optional JSON export."""
    configure(args.dtype)
    setup_logging()

    def solve(D, true_e, true_u, case):
        p, rounds, queries = anchor_tracking_algorithm(
            D, true_e, true_u, config=replace(config, seed=config.seed + case))
        events = DATA["Comparisons"]
        diagnostics = dict(rounds=rounds, comparison_requests=len(events),
                           cache_hits=sum(event["cached"] for event in events),
                           returned_index=p, termination=DATA["Termination"],
                           information_selections=sum(x.get("reason") == "best_item_information_gain"
                                                      for x in DATA["C Decisions"]),
                           selection_fallbacks=sum(x.get("fallback", False) for x in DATA["C Decisions"]))
        return [p], queries, diagnostics

    return run_experiments(args, "AT", DATA, solve, TORCH_DTYPE, output_dir=output_dir)


if __name__ == "__main__":
    from run import parse_args
    main(parse_args().parse_args())
