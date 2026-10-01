"""Joint feedback restoration. Only points and observed answers enter this module.

No work is performed at import time. All solves use float64; returned anchors
are checked again after conversion to the requested execution precision.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class RecoveryConfig:
    margin: float = 1e-6
    tolerance: float = 1e-6
    solver_tolerance: float = 1e-9
    qp_ftol: float = 1e-12
    qp_maxiter: int = 500

    def __post_init__(self):
        values = (self.margin, self.tolerance, self.solver_tolerance, self.qp_ftol)
        if not all(np.isfinite(x) and x > 0 for x in values) or self.qp_maxiter < 1:
            raise ValueError('Positive, finite recovery settings required')
        if self.solver_tolerance < 1e-10:
            raise ValueError('HiGHS feasibility tolerance must be at least 1e-10')

    @classmethod
    def for_dtype(cls, dtype):
        if dtype not in ('float32', 'float64'):
            raise ValueError('Unknown dtype')
        return cls(margin=1e-8 if dtype == 'float64' else 1e-6,
                   tolerance=1e-9 if dtype == 'float64' else 1e-6)


class FeasibilityFailure(RuntimeError):
    def __init__(self, status, **details):
        super().__init__(status)
        self.status = status
        self.details = details


def normalized(A, b):
    scale = np.maximum(np.linalg.norm(A, axis=1), 1e-300)
    return A / scale[:, None], b / scale


def independent_equalities(A, b, tolerance):
    """Remove redundant tie rows for SLSQP; keep all originals for validation."""
    U, singular, V = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(singular > max(A.shape) * np.finfo(float).eps * singular[0]))
    coefficients = U[:, :rank].T @ b
    if np.max(np.abs(b - U[:, :rank] @ coefficients), initial=0) > tolerance:
        raise FeasibilityFailure('inconsistent_equalities')
    return V[:rank], coefficients / singular[:rank]


class FeedbackPolytope:
    """The same bounded polytope is used for recovery, learning, sampling and LP.

    Strict comparisons retain their direction. A common raw-score margin is
    reduced only when a maximum-margin LP proves the requested margin too big.
    A merely weakly feasible state is never accepted as a strict solution.
    """
    def __init__(self, points, records, config=None, dtype='float32'):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or not self.points.size or not np.isfinite(self.points).all():
            raise ValueError('Finite, nonempty point matrix required')
        self.n, self.d = self.points.shape
        self.records = list(records)
        self.config = config or RecoveryConfig.for_dtype(dtype)
        self.dtype = np.dtype(dtype)
        self.native_points = self.points.astype(self.dtype)
        self.lower, self.upper = self.points.min(axis=0), self.points.max(axis=0)
        strict, ties = [], []
        self.strict_pairs, self.tie_pairs = [], []
        for record in records:
            left, right = int(record['left']), int(record['right'])
            if not (0 <= left < self.n and 0 <= right < self.n):
                raise ValueError('Feedback index outside dataset')
            relation = record['relation']
            if relation == 'tie':
                ties.append(self.row(left, right))
                self.tie_pairs.append((left, right))
            elif relation == 'strict':
                strict.append(self.row(left, right))  # left is the observed winner
                self.strict_pairs.append((left, right))
            else:
                raise ValueError('Unknown feedback relation')
        self.H = np.asarray(strict, dtype=float).reshape(-1, 2*self.d)
        self.T = np.asarray(ties, dtype=float).reshape(-1, 2*self.d)
        I = np.eye(self.d)
        self.G = np.vstack((np.c_[-I, np.zeros_like(I)], np.c_[self.lower[:, None]*I, -I],
                            np.c_[-self.upper[:, None]*I, I]))
        self.bounds = [(0., 1.)]*self.d + [
            (min(0., lo), max(0., hi)) for lo, hi in zip(self.lower, self.upper)]
        self.E_full = np.vstack((np.r_[np.ones(self.d), np.zeros(self.d)], self.T))
        self.f_full = np.r_[1., np.zeros(len(self.T))]
        self.E, self.f = independent_equalities(
            self.E_full, self.f_full, self.config.solver_tolerance)
        self.margin, self.witness, self.margin_info = self._find_margin()
        self.A_raw = np.vstack((self.G, self.H))
        self.b_raw = np.r_[np.zeros(len(self.G)), -self.margin*np.ones(len(self.H))]
        self.A, self.b = normalized(self.A_raw, self.b_raw)

    def row(self, winner, loser):
        p, q = self.points[winner], self.points[loser]
        return np.r_[p*p-q*q, -2*(p-q)]

    def _lp(self, c, A, b, E, f, bounds):
        A, b = normalized(A, b)
        return linprog(c, A_ub=A, b_ub=b, A_eq=E, b_eq=f, bounds=bounds,
                       method='highs', options={
                           'primal_feasibility_tolerance': self.config.solver_tolerance,
                           'dual_feasibility_tolerance': self.config.solver_tolerance})

    def _find_margin(self):
        # rho=0 tests the closed/weak region. Maximizing rho diagnoses whether
        # strict comparisons can all hold, instead of declaring weak ties valid.
        A = np.vstack((np.c_[self.G, np.zeros(len(self.G))],
                       np.c_[self.H, np.ones(len(self.H))]))
        result = self._lp(np.r_[np.zeros(2*self.d), -1.], A, np.zeros(len(A)),
                          np.c_[self.E, np.zeros(len(self.E))], self.f,
                          self.bounds + [(0., 1.)])
        if not result.success:
            status = 'feedback_infeasible' if result.status == 2 else 'feasibility_lp_failed'
            raise FeasibilityFailure(status, solver_status=int(result.status), message=str(result.message))
        theta, maximum = result.x[:-1], float(result.x[-1])
        if len(self.H) and maximum <= 10*self.config.solver_tolerance:
            raise FeasibilityFailure('strict_margin_unresolved', maximum_margin=maximum,
                                     weak_feasible=True)
        margin = min(self.config.margin, maximum/4) if len(self.H) else 0.
        return margin, theta, dict(requested=self.config.margin, used=margin,
                                  maximum=maximum, reduced=bool(margin < self.config.margin and len(self.H)))

    def solve_linear(self, objective):
        result = self._lp(objective, self.A_raw, self.b_raw, self.E, self.f, self.bounds)
        if not result.success:
            raise FeasibilityFailure('challenger_lp_failed', solver_status=int(result.status),
                                     message=str(result.message))
        if not self.check_theta(result.x)['valid']:
            raise FeasibilityFailure('lp_solution_failed_validation')
        return result

    def check_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2*self.d,) or not np.isfinite(theta).all():
            return dict(valid=False, reason='nonfinite_or_invalid_shape')
        u = theta[:self.d]
        gaps = self.H @ theta
        ties = self.T @ theta
        bound_violation = max(0., float(np.max(self.G @ theta, initial=0)),
                              float(-u.min()), float(u.max()-1), float(abs(u.sum()-1)))
        feedback_violation = max(0., float(np.max(gaps+self.margin, initial=0)))
        tie_violation = float(np.max(np.abs(ties), initial=0))
        # Positive weights and strict signs are checked independently of tolerance.
        valid = (np.all(u >= 0) and np.all(gaps < 0)
                 and max(bound_violation, feedback_violation, tie_violation) <= self.config.tolerance)
        return dict(valid=bool(valid), max_feedback_gap=float(np.max(gaps)) if len(gaps) else None,
                    max_feedback_violation=feedback_violation, max_domain_violation=bound_violation,
                    max_tie_residual=tie_violation)

    def inspect(self, e, u):
        e, u = np.asarray(e, dtype=float), np.asarray(u, dtype=float)
        if e.shape != (self.d,) or u.shape != (self.d,):
            return dict(valid=False, reason='invalid_shape')
        check = self.check_theta(np.r_[u, u*e])
        if not np.isfinite(e).all():
            return dict(valid=False, reason='nonfinite_anchor')
        outside = float(max(np.max(self.lower-e, initial=0), np.max(e-self.upper, initial=0)))
        check['max_e_bound_violation'] = outside
        check['valid'] = bool(check['valid'] and outside <= self.config.tolerance)
        # Also check the subtraction/reduction actually used by the running model.
        p = self.native_points
        ec, uc = e.astype(self.dtype), u.astype(self.dtype)
        if self.strict_pairs:
            winners, losers = np.asarray(self.strict_pairs, dtype=int).T
            delta = np.sum(uc*(p[winners]-ec)**2, axis=1) - np.sum(uc*(p[losers]-ec)**2, axis=1)
            check['native_strict_valid'] = bool(np.all(delta < 0))
            check['valid'] = bool(check['valid'] and check['native_strict_valid'])
        return check

    def decode(self, theta, reference_e):
        u, z = np.asarray(theta[:self.d]), np.asarray(theta[self.d:])
        e = np.asarray(reference_e, dtype=float).copy()
        positive = u > 0
        e[positive] = z[positive] / u[positive]
        # When u=0, e is unidentifiable. This choice cannot affect the model.
        e[~positive] = np.clip(e[~positive], self.lower[~positive], self.upper[~positive])
        e, u = e.astype(self.dtype), u.astype(self.dtype)
        if self.inspect(e, u)['valid']:
            return e, u
        return None

    def restore(self, e, u):
        """Project jointly; a validated feasible witness is the only fallback."""
        before = self.inspect(e, u)
        if before['valid']:
            return np.asarray(e).copy(), np.asarray(u).copy(), dict(
                triggered=False, status='already_feasible', before=before, after=before,
                margin=self.margin_info, displacement=0.)
        reference = np.r_[np.asarray(u, dtype=float), np.asarray(u, dtype=float)*e]
        constraints = [dict(type='ineq', fun=lambda x: self.b-self.A@x, jac=lambda x: -self.A),
                       dict(type='eq', fun=lambda x: self.E@x-self.f, jac=lambda x: self.E)]
        try:
            result = minimize(lambda x: .5*np.sum((x-reference)**2), self.witness,
                              jac=lambda x: x-reference, method='SLSQP', bounds=self.bounds,
                              constraints=constraints, options={'ftol': self.config.qp_ftol,
                                  'maxiter': self.config.qp_maxiter, 'disp': False})
            qp_success, qp_message = bool(result.success), str(result.message)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            result = None
            qp_success, qp_message = False, f'{type(error).__name__}: {error}'
        candidates = []
        if qp_success and self.check_theta(result.x)['valid']:
            candidates.append(('projection', result.x))
            for weight in (1e-4, 1e-3, .01, .1):
                candidates.append(('projection_roundoff_blend', (1-weight)*result.x+weight*self.witness))
        candidates.append(('feasible_lp_fallback', self.witness))
        for status, theta in candidates:
            decoded = self.decode(theta, e)
            if decoded is None:
                continue
            en, un = decoded
            return en, un, dict(triggered=True, status=status, before=before, after=self.inspect(en, un),
                margin=self.margin_info, qp_success=qp_success, qp_message=qp_message,
                displacement=float(np.linalg.norm(np.r_[un, un*en]-reference)))
        raise FeasibilityFailure('recovery_representation_failed', before=before,
                                 margin=self.margin_info, qp_message=qp_message)

    def description(self):
        return dict(config=asdict(self.config), margin=self.margin_info,
                    domain='per-coordinate observed-data bounds; u>=0; sum(u)=1',
                    strict_count=len(self.H), tie_count=len(self.T))
