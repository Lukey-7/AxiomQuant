#!/usr/bin/env python3
"""Cross-validate the C++ portfolio optimizers against independent reference solutions.

    pip install numpy cvxpy clarabel
    python3 scripts/check_optimizer.py --binary build/bin/axiom_optimizer_dump

For a set of problems (covariances estimated from the bundled sample data, random well-conditioned
matrices, and deliberately ill-conditioned ones, each under several weight bounds) the script runs
`axiom_optimizer_dump` and compares every portfolio with:

  unconstrained GMV, tangency, target return   closed-form NumPy linear algebra
  box-constrained risk aversion and GMV        cvxpy + Clarabel
  box-constrained maximum Sharpe               cvxpy + Clarabel on the homogenised convex QP
  equal risk contribution                      cvxpy + Clarabel on the log-barrier formulation
  bounded-simplex projection                   cvxpy + Clarabel

A result passes when every weight is within --tol (default 1e-8) of the reference. Interior-point
references lose accuracy on ill-conditioned covariances, so a constrained result that misses the
tolerance still passes if it is certified at least as accurate as the reference: feasible, with a
KKT (natural) residual no larger than the reference's and an objective no worse. Those rows are marked.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cvxpy as cp
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLARABEL = dict(solver="CLARABEL", tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13,
                tol_ktratio=1e-10, max_iter=500)
GAMMAS = [0.0, 0.01, 0.1, 1.0, 10.0]
BOUNDS = [(0.0, 1.0), (0.0, 0.4), (-0.2, 0.5), (0.02, 0.3)]


# ----------------------------------------------------------------------------------------------- problems

def sample_data_problem() -> tuple[str, np.ndarray, np.ndarray]:
    closes = []
    for path in sorted((REPO / "sample_data").glob("*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        closes.append([float(r["Close"]) for r in rows])
    length = min(len(c) for c in closes)
    prices = np.array([c[:length] for c in closes]).T
    returns = prices[1:] / prices[:-1] - 1.0
    return "sample_data", returns.mean(axis=0) * 252.0, np.cov(returns, rowvar=False) * 252.0


def random_problem(n: int, seed: int, condition: float | None = None) -> tuple[str, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if condition is None:
        factors = rng.normal(size=(n, max(2, n // 2)))
        sigma = factors @ factors.T * 0.01 + np.diag(rng.uniform(0.01, 0.06, n))
        name = f"random_n{n}_s{seed}"
    else:
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        eigen = 0.04 * np.logspace(0.0, -np.log10(condition), n)
        sigma = q @ np.diag(eigen) @ q.T
        name = f"illcond_n{n}_c{condition:g}"
    sigma = 0.5 * (sigma + sigma.T)
    vol = np.sqrt(np.diag(sigma))
    mu = 0.03 + 0.4 * vol + rng.normal(0.0, 0.02, n)
    return name, mu, sigma


# ----------------------------------------------------------------------------------------------- references

def ref_gmv_unconstrained(mu, sigma, rf):
    x = np.linalg.solve(sigma, np.ones(len(mu)))
    return x / x.sum()


def ref_tangency_unconstrained(mu, sigma, rf):
    x = np.linalg.solve(sigma, mu - rf)
    return x / x.sum()


def ref_target_return(mu, sigma, target):
    n = len(mu)
    kkt = np.zeros((n + 2, n + 2))
    kkt[:n, :n] = sigma
    kkt[:n, n] = kkt[n, :n] = 1.0
    kkt[:n, n + 1] = kkt[n + 1, :n] = mu
    rhs = np.zeros(n + 2)
    rhs[n], rhs[n + 1] = 1.0, target
    return np.linalg.solve(kkt, rhs)[:n]


def solve(problem: cp.Problem) -> None:
    problem.solve(**CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"reference solver status {problem.status}")


def ref_risk_aversion(mu, sigma, gamma, lo, hi):
    w = cp.Variable(len(mu))
    solve(cp.Problem(cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(sigma)) - gamma * mu @ w),
                     [cp.sum(w) == 1, w >= lo, w <= hi]))
    return w.value


def ref_max_sharpe(mu, sigma, rf, lo, hi):
    # Homogenised: y = kappa * w, minimise y' Sigma y subject to (mu - rf)' y = 1.
    y, kappa = cp.Variable(len(mu)), cp.Variable()
    solve(cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))),
                     [(mu - rf) @ y == 1, cp.sum(y) == kappa, y >= lo * kappa, y <= hi * kappa, kappa >= 0]))
    return y.value / kappa.value


def ref_risk_parity(sigma):
    n = sigma.shape[0]
    y = cp.Variable(n)
    solve(cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(sigma)) - cp.sum(cp.log(y)) / n)))
    return y.value / y.value.sum()


def ref_projection(v, lo, hi):
    w = cp.Variable(len(v))
    solve(cp.Problem(cp.Minimize(cp.sum_squares(w - v)), [cp.sum(w) == 1, w >= lo, w <= hi]))
    return w.value


# ----------------------------------------------------------------------------------------------- certificates

def project_bounded_simplex(v, lo, hi):
    a, b = float(np.min(v)) - hi - 1.0, float(np.max(v)) - lo + 1.0
    for _ in range(200):
        t = 0.5 * (a + b)
        if np.clip(v - t, lo, hi).sum() > 1.0:
            a = t
        else:
            b = t
    return np.clip(v - 0.5 * (a + b), lo, hi)


def feasible(w, lo, hi, slack=1e-12):
    return abs(w.sum() - 1.0) <= slack and w.min() >= lo - slack and w.max() <= hi + slack


def qp_certificate(mu, sigma, gamma, lo, hi):
    """Residual and objective of min 0.5 w'Sw - gamma mu'w on the bounded simplex."""
    def residual(w):
        return float(np.max(np.abs(w - project_bounded_simplex(w - (sigma @ w - gamma * mu), lo, hi))))

    def objective(w):
        return float(0.5 * w @ sigma @ w - gamma * mu @ w)

    return residual, objective


# ----------------------------------------------------------------------------------------------- driver

def run_binary(binary: Path, mu, sigma, rf, lo, hi, targets, projections) -> dict[str, np.ndarray]:
    lines = [str(len(mu)), " ".join(f"{x:.17g}" for x in mu)]
    lines += [" ".join(f"{x:.17g}" for x in row) for row in sigma]
    lines.append(f"{rf:.17g} {lo:.17g} {hi:.17g}")
    lines.append(" ".join([str(len(GAMMAS))] + [f"{g:.17g}" for g in GAMMAS]))
    lines.append(" ".join([str(len(targets))] + [f"{t:.17g}" for t in targets]))
    lines.append(str(len(projections)))
    lines += [" ".join(f"{x:.17g}" for x in v) for v in projections]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
        path = handle.name
    try:
        command = [sys.executable, str(binary)] if binary.suffix == ".py" else [str(binary)]
        out = subprocess.run(command + [path], capture_output=True, text=True, check=True).stdout
    finally:
        Path(path).unlink(missing_ok=True)
    results = {}
    for line in out.splitlines():
        name, *values = line.split()
        results[name] = np.array([float(v) for v in values])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=str(REPO / "build" / "bin" / "axiom_optimizer_dump"))
    parser.add_argument("--tol", type=float, default=1e-8, help="maximum absolute weight difference")
    parser.add_argument("--rf", type=float, default=0.02)
    args = parser.parse_args()
    binary = Path(args.binary)

    problems = [sample_data_problem(), random_problem(5, 1), random_problem(10, 2), random_problem(25, 3),
                random_problem(10, 4, condition=1e3), random_problem(10, 5, condition=1e5)]

    rows: list[tuple[str, str, float, str]] = []
    failures: list[str] = []

    def compare(problem: str, name: str, got: np.ndarray | None, expected: np.ndarray, certify=None) -> None:
        """certify(got, expected) -> (passes, note) is consulted only when the weights miss the tolerance."""
        if got is None or got.shape != expected.shape or not np.all(np.isfinite(got)):
            failures.append(f"{problem}: {name} missing or malformed in output")
            return
        err = float(np.max(np.abs(got - expected)))
        note = ""
        if not err <= args.tol:
            passed, note = certify(got, expected) if certify else (False, "")
            if not passed:
                failures.append(f"{problem}: {name} differs by {err:.3e} {note}".rstrip())
        rows.append((problem, name, err, note))

    def certify_qp(mu, sigma, gamma, lo, hi):
        residual, objective = qp_certificate(mu, sigma, gamma, lo, hi)

        def check(got, expected):
            r_got, r_ref = residual(got), residual(expected)
            f_got, f_ref = objective(got), objective(expected)
            ok = feasible(got, lo, hi) and r_got <= max(r_ref, 1e-12) and f_got <= f_ref + 1e-14 * max(1.0, abs(f_ref))
            if ok:
                return True, f"(certified: residual {r_got:.1e} vs reference {r_ref:.1e})"
            return False, f"(residual {r_got:.1e} vs reference {r_ref:.1e}, objective gap {f_got - f_ref:.1e})"
        return check

    def certify_sharpe(mu, sigma, rf, lo, hi):
        def sharpe(w):
            return float((mu @ w - rf) / np.sqrt(w @ sigma @ w))

        def check(got, expected):
            s_got, s_ref = sharpe(got), sharpe(expected)
            ok = feasible(got, lo, hi) and s_got >= s_ref - 1e-13
            if ok:
                return True, f"(certified: Sharpe {s_got:.15f} vs reference {s_ref:.15f})"
            return False, f"(Sharpe {s_got:.15f} vs reference {s_ref:.15f})"
        return check

    for name, mu, sigma in problems:
        n = len(mu)
        rng = np.random.default_rng(n)
        targets = [float(np.min(mu)), float(np.mean(mu)), float(np.max(mu))]
        projections = [rng.normal(0.0, 1.0, n), rng.uniform(-0.1, 0.4, n), np.full(n, 1.0 / n)]

        for lo, hi in BOUNDS:
            if n * lo > 1.0 or n * hi < 1.0:
                continue
            label = f"{name} [{lo:g},{hi:g}]"
            got = run_binary(binary, mu, sigma, args.rf, lo, hi, targets, projections)

            if (lo, hi) == BOUNDS[0]:   # bound-independent results, checked once per problem
                compare(name, "gmv_unconstrained", got.get("gmv_unconstrained"), ref_gmv_unconstrained(mu, sigma, args.rf))
                compare(name, "tangency_unconstrained", got.get("tangency_unconstrained"),
                        ref_tangency_unconstrained(mu, sigma, args.rf))
                for k, target in enumerate(targets):
                    compare(name, f"target_return_{k}", got.get(f"target_return_{k}"), ref_target_return(mu, sigma, target))
                compare(name, "risk_parity", got.get("risk_parity"), ref_risk_parity(sigma))

            compare(label, "gmv_constrained", got.get("gmv_constrained"), ref_risk_aversion(mu, sigma, 0.0, lo, hi),
                    certify_qp(mu, sigma, 0.0, lo, hi))
            for k, gamma in enumerate(GAMMAS):
                compare(label, f"risk_aversion_{k} (gamma={gamma:g})", got.get(f"risk_aversion_{k}"),
                        ref_risk_aversion(mu, sigma, gamma, lo, hi), certify_qp(mu, sigma, gamma, lo, hi))
            if np.max(mu) > args.rf:
                compare(label, "max_sharpe_constrained", got.get("max_sharpe_constrained"),
                        ref_max_sharpe(mu, sigma, args.rf, lo, hi), certify_sharpe(mu, sigma, args.rf, lo, hi))
            for k, v in enumerate(projections):
                # Projection is the QP min 0.5 |w|^2 - v'w, i.e. Sigma = I and gamma * mu = v.
                compare(label, f"projection_{k}", got.get(f"projection_{k}"), ref_projection(v, lo, hi),
                        certify_qp(v, np.eye(n), 1.0, lo, hi))

    width = max(len(r[0]) for r in rows)
    print(f"{'problem':<{width}}  {'portfolio':<34} max |w - w_ref|")
    for problem, name, err, note in rows:
        print(f"{problem:<{width}}  {name:<34} {err:.3e}  {note}".rstrip())
    within = [r for r in rows if r[2] <= args.tol]
    certified = [r for r in rows if r[2] > args.tol and r[3].startswith("(certified")]
    worst = max(within, key=lambda r: r[2]) if within else ("-", "-", float("nan"), "")
    print(f"\n{len(rows)} comparisons: {len(within)} within {args.tol:g} of the reference "
          f"(worst {worst[2]:.3e}, {worst[0]}: {worst[1]}), {len(certified)} certified at least as accurate "
          f"as the reference, {len(failures)} failed")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
