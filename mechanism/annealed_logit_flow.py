from __future__ import annotations

"""
Annealed penalized gradient flow in logits for a reduced, parameter-only RHM model.

This module implements a simple and fast annealed approximation of next-token
prediction with a Zipf-distributed bottom layer. It does NOT enumerate a frozen
RHM instance. Instead, it builds a reduced set of annealed context classes from
(m, v, s, L, a) and evolves the logits of those classes under penalized
population gradient flow:

    dz/dt = p - softmax(z) - lambda * z

in the zero-sum gauge.

Main outputs:
- loss vs norm
- error vs norm (top-1 and soft proxy)
- (ell, k) learned vs norm through a threshold on rho_{ell,k}(t), the weighted
  fraction of ambiguous classes in sector (ell,k) whose correct-token margin is
  positive.

The annealed closure uses a Poisson-distributed number of competitors at each
level ell, with mean derived from the average compatible-set size. This is a
fast parameter-only closure, not an exact annealed solution of the full RHM.

Notebook usage:
    import annealed_logit_flow as alf
    system = alf.build_annealed_classes()
    results = alf.simulate_penalized_gradient_flow(system)
    fig, axs = alf.plot_three_requested(results)

Terminal usage:
    python annealed_logit_flow.py --steps 2000 --lambda_reg 0.5 --save_prefix run1
"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Core helpers
# =========================

def zipf_probs(m: int, a: float) -> np.ndarray:
    """Zipf probabilities f_k ∝ k^{-(1+a)} for k=1,...,m."""
    k = np.arange(1, m + 1, dtype=float)
    w = k ** (-(1.0 + a))
    return w / w.sum()


def mean_compatible_size(L: int, v: int, m: int, s: int) -> np.ndarray:
    """
    Average compatible-set size \bar n_ell used in the annealed reduction.

    Formula:
        \bar n_ell = 1/(1 - m / v^(s-1)) + v * (m / v^(s-1))^(ell-1)
    """
    f = m / (v ** (s - 1))
    ell = np.arange(1, L + 1, dtype=float)
    return 1.0 / (1.0 - f) + v * (f ** (ell - 1))


def stable_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    zmax = np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z - zmax)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def project_zero_sum(z: np.ndarray) -> np.ndarray:
    """Project each class logit vector to zero-sum gauge."""
    return z - z.mean(axis=1, keepdims=True)


# =========================
# Data containers
# =========================

@dataclass
class AnnealedClassSystem:
    L: int
    a: float
    v: int
    m: int
    s: int
    R: int
    seed: int
    f: np.ndarray
    nbar: np.ndarray
    mu: np.ndarray
    weights: np.ndarray          # (C,)
    target_probs: np.ndarray     # (C, v)
    ell: np.ndarray              # (C,) in {0,...,L-1}
    k: np.ndarray                # (C,) in {0,...,m-1}
    n_comp: np.ndarray           # (C,) number of competitors
    ambiguous_mask: np.ndarray   # (C,)
    sector_id: np.ndarray        # (C,) = ell * m + k


@dataclass
class FlowResults:
    params: dict[str, Any]
    norm: np.ndarray
    loss: np.ndarray
    err_top1: np.ndarray
    err_soft: np.ndarray
    rho: np.ndarray              # (T, L, m)
    mean_margin: np.ndarray      # (T, L, m)
    sector_penalty: np.ndarray   # (T, L, m)
    sector_progress: np.ndarray  # (T, L, m)
    time: np.ndarray


# =========================
# Annealed class builder
# =========================

def build_annealed_classes(
    L: int = 3,
    a: float = 2.0,
    v: int = 32,
    m: int = 8,
    s: int = 2,
    R: int = 32,
    seed: int = 0,
    level_weights: np.ndarray | None = None,
) -> AnnealedClassSystem:
    """
    Build reduced annealed classes.

    For each level ell and true bottom-rule rank k, we sample R ambiguity
    subclasses. The number of competitors is drawn from a Poisson law with mean
        mu_ell ≈ \bar n_ell * m / v^(s-1),
    clipped to [0, v-1]. Competitor ranks are drawn iid from the Zipf weights.

    Each class defines a target law p_c over at most 1 + n_comp active tokens;
    the remaining tokens have target probability zero.
    """
    rng = np.random.default_rng(seed)
    f = zipf_probs(m, a)
    nbar = mean_compatible_size(L, v, m, s)
    mu = np.clip(nbar * (m / (v ** (s - 1))), 0.0, v - 1.0)

    if level_weights is None:
        level_weights = np.full(L, 1.0 / L, dtype=float)
    else:
        level_weights = np.asarray(level_weights, dtype=float)
        if level_weights.shape != (L,):
            raise ValueError(f"level_weights must have shape ({L},)")
        if np.any(level_weights < 0):
            raise ValueError("level_weights must be nonnegative")
        total = level_weights.sum()
        if total <= 0:
            raise ValueError("level_weights must sum to a positive number")
        level_weights = level_weights / total

    weights_list = []
    target_probs_list = []
    ell_list = []
    k_list = []
    ncomp_list = []
    ambiguous_list = []
    sector_list = []

    for ell in range(L):
        n_comp_samples = rng.poisson(mu[ell], size=R)
        n_comp_samples = np.clip(n_comp_samples, 0, v - 1)

        for k in range(m):
            fk = f[k]
            base_weight = level_weights[ell] * fk / R

            for n_comp in n_comp_samples:
                if n_comp == 0:
                    comp_ranks = np.array([], dtype=int)
                else:
                    comp_ranks = rng.choice(np.arange(m), size=n_comp, replace=True, p=f)

                active = np.concatenate(([fk], f[comp_ranks]))
                p_active = active / active.sum()

                p_full = np.zeros(v, dtype=float)
                p_full[: len(p_active)] = p_active

                weights_list.append(base_weight)
                target_probs_list.append(p_full)
                ell_list.append(ell)
                k_list.append(k)
                ncomp_list.append(int(n_comp))
                ambiguous_list.append(bool(n_comp > 0))
                sector_list.append(ell * m + k)

    weights = np.asarray(weights_list, dtype=float)
    weights /= weights.sum()  # normalize exactly

    return AnnealedClassSystem(
        L=L,
        a=a,
        v=v,
        m=m,
        s=s,
        R=R,
        seed=seed,
        f=f,
        nbar=nbar,
        mu=mu,
        weights=weights,
        target_probs=np.asarray(target_probs_list, dtype=float),
        ell=np.asarray(ell_list, dtype=int),
        k=np.asarray(k_list, dtype=int),
        n_comp=np.asarray(ncomp_list, dtype=int),
        ambiguous_mask=np.asarray(ambiguous_list, dtype=bool),
        sector_id=np.asarray(sector_list, dtype=int),
    )


# =========================
# Observables
# =========================

def compute_loss(weights: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    ce = -np.sum(p * np.log(np.clip(q, 1e-300, None)), axis=1)
    return float(np.dot(weights, ce))


def compute_errors(weights: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    pred = np.argmax(q, axis=1)
    top1 = 1.0 - p[np.arange(len(pred)), pred]
    soft = 1.0 - np.sum(p * q, axis=1)
    return float(np.dot(weights, top1)), float(np.dot(weights, soft))


def compute_norm(weights: np.ndarray, z: np.ndarray) -> float:
    return 0.5 * float(np.dot(weights, np.sum(z * z, axis=1)))


def class_margin(z: np.ndarray) -> np.ndarray:
    """
    Class margin = correct logit minus largest wrong logit.
    Correct token is always index 0 in the reduced class representation.
    """
    correct = z[:, 0]
    wrong_max = np.max(z[:, 1:], axis=1)
    return correct - wrong_max


def aggregate_sector_observables(system: AnnealedClassSystem, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sector observables on ambiguous classes only:
    - rho_{ell,k}: weighted fraction with positive margin
    - mean_margin_{ell,k}: weighted average margin
    - sector_penalty_{ell,k}: weighted average log(1 + exp(-M))

    Output shapes are (L, m).
    """
    L, m = system.L, system.m
    S = L * m

    mask = system.ambiguous_mask
    if not np.any(mask):
        # fallback; should not happen with default settings
        return (
            np.zeros((L, m), dtype=float),
            np.zeros((L, m), dtype=float),
            np.zeros((L, m), dtype=float),
        )

    sector_id = system.sector_id[mask]
    weights = system.weights[mask]
    M = class_margin(z[mask])

    denom = np.bincount(sector_id, weights=weights, minlength=S)
    mean_num = np.bincount(sector_id, weights=weights * M, minlength=S)
    rho_num = np.bincount(sector_id, weights=weights * (M > 0.0), minlength=S)
    pen_num = np.bincount(sector_id, weights=weights * np.log1p(np.exp(-M)), minlength=S)

    mean_margin = np.divide(mean_num, denom, out=np.full(S, np.nan), where=denom > 0)
    rho = np.divide(rho_num, denom, out=np.full(S, np.nan), where=denom > 0)
    sector_penalty = pen_num  # keep weighted contribution to total loss

    return rho.reshape(L, m), mean_margin.reshape(L, m), sector_penalty.reshape(L, m)


def compute_learning_thresholds(results: FlowResults, tau: float = 0.90) -> np.ndarray:
    """Return N_learn[ell,k] as first norm where rho_{ell,k} >= tau."""
    rho = results.rho
    norm = results.norm
    L = results.params["L"]
    m = results.params["m"]
    out = np.full((L, m), np.nan, dtype=float)
    for ell in range(L):
        for k in range(m):
            vals = rho[:, ell, k]
            idx = np.where(np.isfinite(vals) & (vals >= tau))[0]
            if idx.size > 0:
                out[ell, k] = norm[idx[0]]
    return out


# =========================
# Simulation
# =========================

def simulate_penalized_gradient_flow(
    system: AnnealedClassSystem,
    lambda_reg: float = 0.5,
    lr: float = 0.5,
    steps: int = 2000,
    record_every: int = 1,
    tol: float | None = None,
    verbose: bool = False,
) -> FlowResults:
    """
    Simulate penalized gradient flow in logits:

        dz/dt = p - q - lambda * z

    discretized with Euler updates:

        z <- z + lr * (p - q - lambda * z)

    and projected to the zero-sum gauge after each step.
    """
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if record_every <= 0:
        raise ValueError("record_every must be positive")

    p = system.target_probs.copy()
    weights = system.weights
    C, v = p.shape

    z = np.zeros((C, v), dtype=float)

    time_hist = []
    norm_hist = []
    loss_hist = []
    err_top1_hist = []
    err_soft_hist = []
    rho_hist = []
    mean_margin_hist = []
    sector_penalty_hist = []

    def record(step: int, z_now: np.ndarray) -> None:
        q_now = stable_softmax(z_now)
        time_hist.append(step)
        norm_hist.append(compute_norm(weights, z_now))
        loss_hist.append(compute_loss(weights, p, q_now))
        e1, es = compute_errors(weights, p, q_now)
        err_top1_hist.append(e1)
        err_soft_hist.append(es)
        rho, mean_margin, sector_penalty = aggregate_sector_observables(system, z_now)
        rho_hist.append(rho)
        mean_margin_hist.append(mean_margin)
        sector_penalty_hist.append(sector_penalty)

    record(0, z)
    last_loss = loss_hist[-1]

    for step in range(1, steps + 1):
        q = stable_softmax(z)
        grad = p - q - lambda_reg * z
        z = z + lr * grad
        z = project_zero_sum(z)

        if step % record_every == 0 or step == steps:
            record(step, z)
            current_loss = loss_hist[-1]
            if verbose and (step % max(record_every, steps // 10 if steps >= 10 else 1) == 0 or step == steps):
                print(
                    f"step={step:6d}  norm={norm_hist[-1]:.6f}  loss={current_loss:.6f}  err_soft={err_soft_hist[-1]:.6f}"
                )
            if tol is not None and abs(last_loss - current_loss) < tol:
                if verbose:
                    print(f"Converged early at step={step} with |Δloss|<{tol}")
                break
            last_loss = current_loss

    rho_arr = np.asarray(rho_hist, dtype=float)
    mean_margin_arr = np.asarray(mean_margin_hist, dtype=float)
    sector_penalty_arr = np.asarray(sector_penalty_hist, dtype=float)

    # Sector progress from reduction in sector penalty.
    C0 = sector_penalty_arr[0]
    Cinf = sector_penalty_arr[-1]
    denom = np.maximum(C0 - Cinf, 1e-12)
    sector_progress = np.clip((C0[None, :, :] - sector_penalty_arr) / denom[None, :, :], 0.0, 1.0)

    return FlowResults(
        params={
            "L": system.L,
            "a": system.a,
            "v": system.v,
            "m": system.m,
            "s": system.s,
            "R": system.R,
            "seed": system.seed,
            "lambda_reg": lambda_reg,
            "lr": lr,
            "steps": steps,
            "record_every": record_every,
            "tol": tol,
        },
        norm=np.asarray(norm_hist, dtype=float),
        loss=np.asarray(loss_hist, dtype=float),
        err_top1=np.asarray(err_top1_hist, dtype=float),
        err_soft=np.asarray(err_soft_hist, dtype=float),
        rho=rho_arr,
        mean_margin=mean_margin_arr,
        sector_penalty=sector_penalty_arr,
        sector_progress=sector_progress,
        time=np.asarray(time_hist, dtype=float),
    )


# =========================
# Plotting
# =========================

def plot_three_requested(results: FlowResults, tau: float = 0.90, figsize: tuple[float, float] = (17, 4.8)):
    """
    Plot:
    1) loss vs norm
    2) error vs norm
    3) (ell,k) learned vs norm via threshold N_learn on rho_{ell,k}
    """
    L = results.params["L"]
    m = results.params["m"]
    N_learn = compute_learning_thresholds(results, tau=tau)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Loss
    axs[0].plot(results.norm, results.loss, lw=2)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("logit norm")
    axs[0].set_ylabel("population loss")
    axs[0].set_title("Loss vs norm")
    axs[0].grid(True, alpha=0.3)

    # Error
    axs[1].plot(results.norm, results.err_top1, lw=2, label="top-1 error")
    axs[1].plot(results.norm, results.err_soft, lw=2, ls="--", label="soft error")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("logit norm")
    axs[1].set_ylabel("error")
    axs[1].set_title("Error vs norm")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(frameon=False)

    # Learned sectors scatter
    xs = []
    ys = []
    ylabels = []
    colors = []
    for ell in range(L):
        for k in range(m):
            xs.append(N_learn[ell, k])
            ys.append(ell * m + k)
            ylabels.append(f"({ell+1},{k+1})")
            colors.append(ell + 1)

    axs[2].scatter(xs, ys, c=colors, s=50)
    axs[2].set_xscale("log")
    axs[2].set_xlabel(f"norm at rho >= {tau:.2f}")
    axs[2].set_ylabel("(ell, k)")
    axs[2].set_title("(ell,k) learned vs norm")
    axs[2].set_yticks(np.arange(L * m))
    axs[2].set_yticklabels(ylabels, fontsize=8)
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axs


def plot_sector_heatmap(results: FlowResults, observable: str = "rho", figsize: tuple[float, float] = (10, 6)):
    """Plot a heatmap of a sector observable vs norm for all (ell,k)."""
    L = results.params["L"]
    m = results.params["m"]

    if observable == "rho":
        arr = results.rho
        title = r"$\rho_{\ell,k}(N)$"
    elif observable == "mean_margin":
        arr = results.mean_margin
        title = r"$\bar M_{\ell,k}(N)$"
    elif observable == "progress":
        arr = results.sector_progress
        title = r"sector progress"
    else:
        raise ValueError("observable must be one of: 'rho', 'mean_margin', 'progress'")

    mat = arr.reshape(arr.shape[0], L * m).T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("recorded step index")
    ax.set_ylabel("(ell,k)")
    ax.set_yticks(np.arange(L * m))
    ax.set_yticklabels([f"({ell+1},{k+1})" for ell in range(L) for k in range(m)], fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


# =========================
# Saving helpers
# =========================

def save_results_npz(path: str | Path, results: FlowResults) -> None:
    path = Path(path)
    np.savez(
        path,
        norm=results.norm,
        loss=results.loss,
        err_top1=results.err_top1,
        err_soft=results.err_soft,
        rho=results.rho,
        mean_margin=results.mean_margin,
        sector_penalty=results.sector_penalty,
        sector_progress=results.sector_progress,
        time=results.time,
        params=np.array([results.params], dtype=object),
    )


# =========================
# CLI
# =========================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Annealed penalized gradient flow in logits for a reduced RHM model")
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--v", type=int, default=32)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--R", type=int, default=32, help="number of ambiguity subclasses per (ell,k)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lambda_reg", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--record_every", type=int, default=5)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.90, help="rho threshold for learned sectors")

    p.add_argument("--save_prefix", type=str, default=None, help="if given, save plot and npz with this prefix")
    p.add_argument("--show", action="store_true", help="show plots interactively")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    system = build_annealed_classes(
        L=args.L,
        a=args.a,
        v=args.v,
        m=args.m,
        s=args.s,
        R=args.R,
        seed=args.seed,
    )

    results = simulate_penalized_gradient_flow(
        system,
        lambda_reg=args.lambda_reg,
        lr=args.lr,
        steps=args.steps,
        record_every=args.record_every,
        tol=args.tol,
        verbose=args.verbose,
    )

    fig, _ = plot_three_requested(results, tau=args.tau)

    if args.save_prefix is not None:
        prefix = Path(args.save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        save_results_npz(prefix.with_suffix(".npz"), results)
        fig.savefig(prefix.with_name(prefix.name + "_three_plots.png"), dpi=200, bbox_inches="tight")
        print(f"Saved results to {prefix.with_suffix('.npz')}")
        print(f"Saved figure to {prefix.with_name(prefix.name + '_three_plots.png')}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
