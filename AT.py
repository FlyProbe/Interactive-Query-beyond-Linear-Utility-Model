"""Anchor tracking with ACIG selection, joint recovery and stabilized updates.

Select one unseen question, update the anchor, then replan both chains.
Information gain scores questions only; the stopping decision uses the LP.
DATA keeps the existing monitoring schema. Each Time / E-U Route entry is
recorded after one paid question and its anchor update; cached answers add none.
Detailed recovery, chain and learning states are recorded only with debug=True.
"""
import time
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
from at_feasibility import FeedbackPolytope, FeasibilityFailure, RecoveryConfig


# Import shared runtime initialization before importing PyTorch.
from utils import (get_weighted_distance_sq, initialize_monitor, record_query_state,
                   run_experiments, setup_logging, synchronize)

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


def _feedback_records(history):
    """Accept legacy winner/loser tuples as well as explicit tie records."""
    return [dict(record) if isinstance(record, dict) else
            dict(left=int(record[0]), right=int(record[1]), relation='strict')
            for record in history]


def _polytope(D, history, recovery_config=None):
    dtype = 'float64' if D.dtype == torch.float64 else 'float32'
    return FeedbackPolytope(D.detach().cpu().numpy(), _feedback_records(history),
                            recovery_config, dtype)


def at_check_stopping_condition_LP(
        history_constraints, D, p_idx, e, u, visited_rebuttals, device='cpu', *,
        poly=None, recovery_config=None):
    """Find a validated challenger; numerical failures never certify stopping."""
    if poly is None:
        poly = _polytope(D, history_constraints, recovery_config)
    return challenger(poly, D, p_idx, e, u, visited_rebuttals)


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
        D, history_constraints, current_e, current_u, max_iter=500, init_lr=.5,
        device='cpu', *, proximal_strength=1., recovery_config=None, poly=None):
    """Restore all feedback jointly before stabilized alternating descent.

    The proximal reference is the feasible anchor restored for this call.
    Returning the old infeasible anchor is never a recovery fallback.
    """
    if poly is None:
        poly = _polytope(D, history_constraints, recovery_config)
    e, u, _ = restore_anchor(poly, current_e.clone(), current_u.clone(), D)
    return learn(poly, D, e, u, max_iter, init_lr, proximal_strength)


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
    """ACIG hit-and-run with RHS offsets and a null space for explicit ties.

    This is a query-scoring ensemble, not a parameter posterior. Degenerate
    clouds retain the existing information-gain fallback to the rebuttal pair.
    """
    def __init__(self, points, config=SelectionConfig()):
        self.points = np.asarray(points, dtype=np.float64)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.samples = None
        self.last_status = 'not_sampled'

    def draw(self, poly):
        if not isinstance(poly, FeedbackPolytope):
            poly = FeedbackPolytope(self.points, _feedback_records(poly))
        cfg, A, b = self.config, poly.A, poly.b
        _, _, V = np.linalg.svd(poly.E, full_matrices=True)
        basis = V[len(poly.E):].T
        previous = self.samples
        if previous is not None and len(previous):
            okay = np.all(previous@A.T <= b+cfg.tolerance, axis=1)
            okay &= np.all(np.abs(previous@poly.E.T-poly.f) <= cfg.tolerance, axis=1)
            previous = previous[okay]
        if previous is not None and len(previous):
            state = previous.mean(axis=0)
        else:
            radii = np.linalg.norm(A@basis, axis=1)
            result = poly._lp(np.r_[np.zeros(2*poly.d), -1.], np.c_[A, radii], b,
                              np.c_[poly.E, np.zeros(len(poly.E))], poly.f,
                              poly.bounds+[(0., 1.)])
            if not result.success:
                self.samples = np.empty((0, 2*poly.d))
                self.last_status = 'sampler_lp_failed'
                return self.samples
            state = result.x[:-1]
        if not basis.shape[1]:
            if not poly.check_theta(state)['valid']:
                self.samples = np.empty((0, 2*poly.d))
                self.last_status = 'invalid_zero_dimensional_state'
                return self.samples
            self.samples = np.repeat(state[None, :], cfg.samples, axis=0)
            self.last_status = 'zero_dimensional'
            return self.samples
        draws = []
        for step in range(cfg.burn_in+cfg.samples*cfg.thinning):
            direction = basis@self.rng.normal(size=basis.shape[1])
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm
            slack, slopes = b-A@state, A@direction
            if np.min(slack) < -cfg.tolerance:
                break
            slack = np.maximum(slack, 0.)
            negative, positive = slopes < -1e-14, slopes > 1e-14
            low = np.max(slack[negative]/slopes[negative], initial=-np.inf)
            high = np.min(slack[positive]/slopes[positive], initial=np.inf)
            # Enforce the explicit variable bounds as well as the domain rows.
            for coordinate, (lower, upper) in enumerate(poly.bounds):
                slope = direction[coordinate]
                if abs(slope) > 1e-14:
                    ends = ((lower-state[coordinate])/slope, (upper-state[coordinate])/slope)
                    low, high = max(low, min(ends)), min(high, max(ends))
            if not np.isfinite(low+high) or high < low:
                break
            state = state+(low+(high-low)*self.rng.uniform(1e-7, 1-1e-7))*direction
            if step >= cfg.burn_in and (step-cfg.burn_in) % cfg.thinning == 0:
                if poly.check_theta(state)['valid']:
                    draws.append(state.copy())
        self.samples = np.asarray(draws).reshape(-1, 2*poly.d)
        self.last_status = 'ok' if len(draws) == cfg.samples else 'incomplete_cloud'
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


def check_anchor(poly, e, u, D):
    check = poly.inspect(e.detach().cpu().numpy(), u.detach().cpu().numpy())
    # NumPy and CUDA can round reductions differently. Validate both the lifted
    # float64 expression and the actual tensor expression consumed by ACIG.
    for pairs, relation in ((poly.strict_pairs, 'strict'), (poly.tie_pairs, 'tie')):
        if not pairs:
            continue
        indices = torch.tensor(pairs, dtype=torch.long, device=D.device)
        gap = torch.sum(u*(D[indices[:, 0]]-e)**2, dim=1) - torch.sum(u*(D[indices[:, 1]]-e)**2, dim=1)
        valid = torch.all(gap < 0) if relation == 'strict' else torch.all(torch.abs(gap) <= poly.config.tolerance)
        check['tensor_'+relation+'_valid'] = bool(valid.item())
        check['valid'] = bool(check['valid'] and valid.item())
    return check


def restore_anchor(poly, e, u, D):
    start = time.perf_counter()
    before = check_anchor(poly, e, u, D)
    if before['valid']:
        return e, u, dict(triggered=False, status='already_feasible', before=before,
                          after=before, elapsed=time.perf_counter()-start, displacement=0.,
                          margin=poly.margin_info)
    en, un, diagnostic = poly.restore(e.detach().cpu().numpy(), u.detach().cpu().numpy())
    candidates = [(en, un, diagnostic['status'])]
    fallback = poly.decode(poly.witness, e.detach().cpu().numpy())
    if fallback is not None:
        candidates.append((*fallback, 'tensor_feasible_lp_fallback'))
    for ev, uv, status in candidates:
        et = torch.as_tensor(ev, dtype=D.dtype, device=D.device)
        ut = torch.as_tensor(uv, dtype=D.dtype, device=D.device)
        after = check_anchor(poly, et, ut, D)
        if after['valid']:
            old_u, old_e = u.detach().cpu().numpy(), e.detach().cpu().numpy()
            actual_u, actual_e = ut.detach().cpu().numpy(), et.detach().cpu().numpy()
            diagnostic.update(triggered=True, status=status, before=before, after=after,
                              displacement=float(np.linalg.norm(np.r_[actual_u, actual_u*actual_e]-
                                                                 np.r_[old_u, old_u*old_e])),
                              elapsed=time.perf_counter()-start)
            return et, ut, diagnostic
    raise FeasibilityFailure('tensor_recovery_failed', before=before, margin=poly.margin_info)


def learn(poly, D, e, u, max_iter=500, init_lr=.5, proximal_strength=1., trace=None):
    """ACIG alternating descent, optionally stabilized around this round's restored anchor."""
    if not check_anchor(poly, e, u, D)['valid']:
        raise FeasibilityFailure('learning_started_from_invalid_anchor')
    if not poly.strict_pairs:
        return e, u
    pairs = torch.tensor(poly.strict_pairs, dtype=torch.long, device=D.device)
    P, Q = D[pairs[:, 0]], D[pairs[:, 1]]
    lower = torch.as_tensor(poly.lower, dtype=D.dtype, device=D.device)
    upper = torch.as_tensor(poly.upper, dtype=D.dtype, device=D.device)
    reference_e, reference_u = e.clone(), u.clone()
    span = torch.clamp(upper-lower, min=torch.finfo(D.dtype).eps)

    def loss(et, ut):
        fit = F.softplus(torch.sum(ut*(P-et)**2, dim=1)-torch.sum(ut*(Q-et)**2, dim=1)).sum()
        proximity = .5*proximal_strength*(((et-reference_e)/span).square().sum() + (ut-reference_u).square().sum())
        return (fit+proximity).item()

    # Preserve the selected experimental sum-loss / mean-gradient arithmetic.
    for iteration in range(max_iter):
        prev_e, prev_u = e.clone(), u.clone()
        terms = (P-e)**2-(Q-e)**2
        # Match the original arithmetic for the gradient's score difference.
        delta = torch.sum(u*(P-e)**2, dim=1)-torch.sum(u*(Q-e)**2, dim=1)
        grad_u = torch.mean(torch.sigmoid(delta).unsqueeze(-1)*terms, dim=0) + proximal_strength*(u-reference_u)
        previous, rate = loss(e, u), init_lr
        while rate > 1e-5:
            candidate = torch.clamp(u-rate*grad_u, .001, 1.)
            candidate = candidate/candidate.sum()
            if loss(e, candidate) < previous and check_anchor(poly, e, candidate, D)['valid']:
                u = candidate
                if trace is not None:
                    trace(iteration, 'u', e, u, loss(e, u))
                break
            rate *= .5
        delta = torch.sum(u*(P-e)**2, dim=1)-torch.sum(u*(Q-e)**2, dim=1)
        grad_e = torch.mean(torch.sigmoid(delta).unsqueeze(-1)*(2*u*(Q-P)), dim=0) + proximal_strength*(e-reference_e)/span.square()
        previous, rate = loss(e, u), init_lr
        while rate > 1e-5:
            candidate = torch.clamp(e-rate*grad_e, lower, upper)
            if loss(candidate, u) < previous and check_anchor(poly, candidate, u, D)['valid']:
                e = candidate
                if trace is not None:
                    trace(iteration, 'e', e, u, loss(e, u))
                break
            rate *= .5
        if iteration > 50 and torch.linalg.norm(e-prev_e) < 1e-5 and torch.linalg.norm(u-prev_u) < 1e-5:
            break
    return e, u


def challenger(poly, D, p, e, u, known_losers):
    """No LP error/infeasibility is allowed to masquerade as certification."""
    if not check_anchor(poly, e, u, D)['valid']:
        raise FeasibilityFailure('invalid_anchor_before_challenger')
    order = torch.argsort(get_weighted_distance_sq(D, e, u)).cpu().tolist()
    for q in order:
        if q == p or q in known_losers:
            continue
        row = poly.row(q, p)
        result = poly.solve_linear(row)
        if float(result.fun) >= -LP_ACC:
            continue
        # LP vertices can lose a tiny strict feedback margin when cast to
        # float32. Blend toward the verified interior-feedback witness, while
        # retaining at least half the challenger threshold in exact arithmetic.
        minimum, witness_gap = float(result.fun), float(row@poly.witness)
        alpha_limit = 1. if witness_gap <= minimum else min(
            1., max(0., (-LP_ACC/2-minimum)/(witness_gap-minimum)))
        for alpha in (0., min(1e-4, alpha_limit/2), min(.001, alpha_limit/2),
                      min(.01, alpha_limit/2), min(.1, alpha_limit/2), alpha_limit/2):
            theta = (1-alpha)*result.x+alpha*poly.witness
            decoded = poly.decode(theta, e.detach().cpu().numpy())
            if decoded is None:
                continue
            ep, up = [torch.as_tensor(x, dtype=D.dtype, device=D.device) for x in decoded]
            tensor_gap = get_weighted_distance_sq(D[q], ep, up)-get_weighted_distance_sq(D[p], ep, up)
            if check_anchor(poly, ep, up, D)['valid'] and tensor_gap.item() < 0:
                return q, ep, up
        raise FeasibilityFailure('challenger_tensor_validation_failed', challenger=q)
    return None, None, None


class _ATRecorder:
    """Keep query-aligned plot data; detailed intermediate states are opt-in."""
    def __init__(self, D, true_e, true_u, debug):
        self.D, self.debug = D, bool(debug)
        self.started = time.perf_counter()
        self.extra_seconds = 0.
        self.data = DATA
        initialize_monitor(self.data, D, true_e, true_u)
        self.data.update({
            'Schema Version': 2, 'Algorithm': 'AT',
            'State Semantics': 'feedback_feasible_operational_anchor',
            'Debug Enabled': self.debug, 'Feedback': [], 'Query Route': [],
            'Recommendation Route': [], 'Phase Route': [],
            'Monitoring Notes': 'Time includes monitoring; true parameters only answer queries and evaluate results.',
        })
        if self.debug:
            self.data.update({'State Events': [], 'Recovery Diagnostics': []})

    def extra(self, operation):
        synchronize(self.D.device)
        started = time.perf_counter()
        result = operation()
        synchronize(self.D.device)
        self.extra_seconds += time.perf_counter()-started
        return result

    def event(self, queries, e, u, indices, phase, *, anchor_valid=True, **details):
        if not self.debug:
            return
        def append():
            self.data['State Events'].append(dict(
                queries=int(queries), phase=phase, e_u=[e.tolist(), u.tolist()],
                recommendation=list(map(int, indices)),
                anchor_valid=None if anchor_valid is None else bool(anchor_valid), **details))
        self.extra(append)

    def initial(self, e, u, p):
        self.data['Initial E-U'] = [e.tolist(), u.tolist()]
        self.data['Initial Recommendation'] = [int(p)]
        self.event(0, e, u, [p], 'initial', anchor_valid=None)

    def recovery(self, queries, phase, diagnostic, **details):
        if self.debug:
            self.data['Recovery Diagnostics'].append(dict(
                queries=int(queries), phase=phase, **diagnostic, **details))

    def paid(self, e, u, p, phase, valid=True):
        record_query_state(self.data, e, u, self.started)
        queries = len(self.data['E-U Route'])
        self.data['Query Route'].append(queries)
        self.data['Recommendation Route'].append([int(p)])
        self.data['Phase Route'].append(phase)
        self.data['Anchor Validity Route'].append(bool(valid))
        self.event(queries, e, u, [p], phase, anchor_valid=valid)

    def finish(self, e, u, p, reason, rounds, queries, valid):
        self.event(queries, e, u, [p], 'terminal:'+reason, anchor_valid=valid)
        self.data.update({'Final E-U': [e.tolist(), u.tolist()],
                          'Final Recommendation': [int(p)], 'Termination': reason,
                          'Rounds': int(rounds), 'Queries': int(queries),
                          'Final Anchor Valid': bool(valid), 'Algorithm Valid': bool(valid)})
        synchronize(self.D.device)
        self.data['Final Time'] = time.perf_counter()-self.started
        self.data['Monitoring Time'] = self.extra_seconds
        assert queries == len(self.data['Time']) == len(self.data['E-U Route'])
        assert queries == len(self.data['Recommendation Route']) == len(self.data['Feedback'])
        assert queries == len(self.data['Feedback Records']) == len(self.data['Anchor Validity Route'])
        assert queries == sum(event['cost'] for event in self.data['Comparisons'])
        return int(p), int(rounds), int(queries)


def anchor_tracking_algorithm(D, true_e, true_u, max_round=500,
                              config=SelectionConfig(), initial_anchor=None, *,
                              debug=False, recovery_config=None, proximal_strength=1.,
                              query_budget=None):
    """ACIG with joint recovery and per-query proximal alternating updates.

    Native random initialization is the default used by the experiment runner.
    initial_anchor remains an explicit programmatic hook for controlled tests.
    LP jumps retain the selected experiment's direct feasible rebuttal anchor.
    max_round includes zero-query relocations; debug only adds observations.
    """
    if D.dtype not in (torch.float32, torch.float64):
        raise ValueError('AT requires float32 or float64 data')
    if max_round < 0 or (query_budget is not None and query_budget < 0):
        raise ValueError('Invalid query/round cap')
    if not np.isfinite(proximal_strength) or proximal_strength < 0:
        raise ValueError('proximal_strength must be finite and nonnegative')
    dtype = 'float64' if D.dtype == torch.float64 else 'float32'
    configure(dtype)
    recovery_config = recovery_config or RecoveryConfig.for_dtype(dtype)
    recorder = _ATRecorder(D, true_e, true_u, debug)
    data = recorder.data
    if initial_anchor is None:
        lower, upper = D.min(dim=0).values, D.max(dim=0).values
        # Preserve native ACIG draws, including their default tensor precision.
        e = lower+torch.rand(D.shape[1], device=D.device)*(upper-lower)
        u = torch.distributions.Dirichlet(torch.ones(D.shape[1], device=D.device)).sample()
    else:
        e, u = [torch.as_tensor(x, dtype=D.dtype, device=D.device).clone() for x in initial_anchor]
    cache = ComparisonCache(D, true_e, true_u)
    records = []
    data.update({'Initialization': 'random' if initial_anchor is None else 'supplied',
                 'Comparisons': cache.events, 'C Decisions': [], 'Feedback Records': records,
                 'Anchor Validity Route': [], 'Feasibility Config': vars(recovery_config),
                 'Variant': 'AT_joint_recovery' if proximal_strength == 0 else 'AT_joint_stable',
                 'Proximal Strength': float(proximal_strength),
                 'Proximal Reference': 'per-query restored feasible anchor'})
    points = D.detach().cpu().numpy().astype(np.float64)
    sampler = FeasibleSamples(points, config)
    poly, rounds = None, 0
    p, _ = at_find_nearest_neighbor(D, e, u)
    recorder.initial(e, u, p)

    def finish(reason, failure=None):
        final_p, _ = at_find_nearest_neighbor(D, e, u)
        valid = failure is None and poly is not None and check_anchor(poly, e, u, D)['valid']
        if failure is not None:
            data['Failure'] = dict(status=failure.status, **failure.details)
        return recorder.finish(e, u, final_p, reason, rounds, cache.queries_used, valid)

    try:
        poly = FeedbackPolytope(points, records, recovery_config, dtype)
        e, u, initial_diag = restore_anchor(poly, e, u, D)
        if debug:
            data['State Events'][0]['anchor_valid'] = initial_diag['before']['valid']
        if initial_diag['triggered']:
            recorder.recovery(0, 'initial', initial_diag)
            initial_p, _ = at_find_nearest_neighbor(D, e, u)
            recorder.event(0, e, u, [initial_p], 'initial_recovery')
        while rounds < max_round:
            p, _ = at_find_nearest_neighbor(D, e, u)
            known_losers = {loser for winner, loser in poly.strict_pairs if winner == p}
            known_losers.update(right if left == p else left
                               for left, right in poly.tie_pairs if p in (left, right))
            q, ep, up = at_check_stopping_condition_LP(
                records, D, p, e, u, known_losers, device=D.device, poly=poly)
            if q is None:
                data['Certification'] = dict(kind='LP_within_configured_margin_and_tolerance',
                                              **poly.description())
                return finish('LP')
            if debug:
                rebuttal_p, _ = at_find_nearest_neighbor(D, ep, up)
                recorder.event(cache.queries_used, ep, up, [rebuttal_p], 'lp_rebuttal',
                               incumbent=p, rebuttal=q)
            key = tuple(sorted((p, q)))
            if key in cache.answers:
                prefers_p, _ = cache.compare(p, q, 'cached_rebuttal')
                if prefers_p or cache.answers[key] == 0:
                    raise FeasibilityFailure('cached_rebuttal_conflict')
                e, u, diagnostic = restore_anchor(poly, ep, up, D)
                recorder.recovery(cache.queries_used, 'cached_relocation', diagnostic)
                data['C Decisions'].append(dict(round=rounds, action='cached_relocation', pair=[p, q]))
                new_p, _ = at_find_nearest_neighbor(D, e, u)
                recorder.event(cache.queries_used, e, u, [new_p], 'cached_relocation')
                rounds += 1
                continue
            if query_budget is not None and cache.queries_used >= query_budget:
                return finish('query_budget')
            u_chain, u_states = at_construct_q_chain_alternating(D, e, u, ep, up, True)
            e_chain, e_states = at_construct_q_chain_alternating(D, e, u, ep, up, False)
            candidates = candidate_pairs(u_chain, e_chain, p, q, cache.answers)
            samples = sampler.draw(poly)
            pair, info = select_pair(points, samples, candidates, config.tolerance)
            if pair is None:
                raise FeasibilityFailure('no_unseen_candidate')
            left, right = pair
            prefers_left, cost = cache.compare(left, right, 'C:'+'+'.join(info['sources']))
            if cost != 1:
                raise FeasibilityFailure('selected_cached_question')
            winner, loser = (left, right) if prefers_left else (right, left)
            tied = cache.answers[tuple(sorted(pair))] == 0
            data['Feedback'].append((winner, loser))  # one legacy entry per paid question
            records.append(dict(left=left if tied else winner, right=right if tied else loser,
                                relation='tie' if tied else 'strict'))
            poly = FeedbackPolytope(points, records, recovery_config, dtype)
            if debug:
                recorder.event(cache.queries_used, e, u, [p], 'feedback_received',
                               anchor_valid=check_anchor(poly, e, u, D)['valid'],
                               pair=list(pair), tied=tied)
            if pair == key and winner == q and not tied:
                # Keep the LP jump. No landing optimization or extra descent is added.
                e, u, diagnostic = restore_anchor(poly, ep, up, D)
                update = 'A_rebuttal_jump'
            else:
                e, u, diagnostic = restore_anchor(poly, e, u, D)
                restored_e, restored_u = e.clone(), u.clone()
                if debug:
                    restored_p, _ = at_find_nearest_neighbor(D, e, u)
                    recorder.event(cache.queries_used, e, u, [restored_p], 'joint_recovery')
                    diagnostic['proximal_reference_e_u'] = [e.tolist(), u.tolist()]

                def trace(iteration, coordinate, et, ut, loss):
                    current_p, _ = at_find_nearest_neighbor(D, et, ut)
                    recorder.event(cache.queries_used, et, ut, [current_p], 'learning_'+coordinate,
                                   iteration=int(iteration), loss=float(loss))

                e, u = learn(poly, D, e, u, proximal_strength=proximal_strength,
                             trace=trace if debug else None)
                if not check_anchor(poly, e, u, D)['valid']:
                    e, u = restored_e, restored_u
                    diagnostic['learning_fallback'] = 'restored_anchor'
                update = ('A_joint_recovery_then_alternating' if proximal_strength == 0 else
                          'A_joint_recovery_then_proximal_alternating')
            after = check_anchor(poly, e, u, D)
            if not after['valid']:
                raise FeasibilityFailure('invalid_anchor_before_record', check=after)
            recorder.recovery(cache.queries_used, update, diagnostic, final_check=after)
            current_p = recorder.extra(lambda: at_find_nearest_neighbor(D, e, u)[0])
            recorder.paid(e, u, current_p, update)
            decision = dict(round=rounds, action='ask_one', pair=list(pair),
                winner=None if tied else winner, tied=tied, incumbent=p, rebuttal=q,
                u_chain_length=len(u_chain), e_chain_length=len(e_chain), candidate_count=len(candidates),
                sampler_status=sampler.last_status, anchor_update=update, **info)
            if debug:
                decision.update(u_chain=list(u_chain), e_chain=list(e_chain),
                                u_chain_e_u=[[et.tolist(), ut.tolist()] for et, ut in u_states],
                                e_chain_e_u=[[et.tolist(), ut.tolist()] for et, ut in e_states],
                                candidates=[dict(pair=list(k), sources=v) for k, v in candidates.items()],
                                rebuttal_e_u=[ep.tolist(), up.tolist()])
            data['C Decisions'].append(decision)
            rounds += 1
        return finish('max_round')
    except FeasibilityFailure as failure:
        # Account for a paid query even when its new feedback cannot be restored.
        # Mark the old anchor invalid rather than accepting it as a solution.
        if cache.queries_used > len(data['E-U Route']):
            current_p, _ = at_find_nearest_neighbor(D, e, u)
            recorder.paid(e, u, current_p, 'invalid_state:'+failure.status, valid=False)
        return finish('feasibility_failure', failure)


def main(args, output_dir=None, config=SelectionConfig()):
    """Run independent AT trials with shared monitoring and optional JSON export."""
    configure(args.dtype)
    setup_logging()

    def solve(D, true_e, true_u, case):
        p, rounds, queries = anchor_tracking_algorithm(
            D, true_e, true_u, config=replace(config, seed=config.seed + case),
            debug=getattr(args, "debug", False))
        events = DATA["Comparisons"]
        diagnostics = dict(rounds=rounds, comparison_requests=len(events),
                           cache_hits=sum(event["cached"] for event in events),
                           returned_index=p, termination=DATA["Termination"],
                           algorithm_valid=DATA["Algorithm Valid"],
                           final_anchor_valid=DATA["Final Anchor Valid"],
                           information_selections=sum(x.get("reason") == "best_item_information_gain"
                                                      for x in DATA["C Decisions"]),
                           selection_fallbacks=sum(x.get("fallback", False) for x in DATA["C Decisions"]))
        return [p], queries, diagnostics

    return run_experiments(args, "AT", DATA, solve, TORCH_DTYPE, output_dir=output_dir)


if __name__ == "__main__":
    from run import parse_args
    main(parse_args().parse_args())
