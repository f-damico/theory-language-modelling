#!/usr/bin/env python3
"""
Plot utilities for constrained_bp_last_next_torch.py outputs.

Can be imported in a notebook or run from terminal:

    python plot_constrained_bp_results.py --input results.npz --out_prefix figures/run1

The functions accept either a path to a saved .npz file or an already-loaded
result dictionary returned by simulate_constrained_bp_sweep(...).

All line-plot functions support independent x/y log scaling via log_x and log_y.
By default both are True. If an axis contains zero or negative finite values, the
function uses a symmetric-log scale instead of an ordinary log scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str | Path) -> Dict[str, Any]:
    """Load the .npz produced by save_results_npz(...)."""
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    if "params_json" in out:
        try:
            out["params"] = json.loads(str(out["params_json"]))
        except Exception:
            out["params"] = {}
    return out


def _as_results(results_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(results_or_path, (str, Path)):
        return load_results(results_or_path)
    return results_or_path


def _extract_x(results: Dict[str, Any], x_key: str) -> Tuple[np.ndarray, str]:
    labels = {
        "lambda_values": r"target message budget $\lambda$",
        "message_total_cost_mean": r"measured message cost $\langle\sum_e\|c_e\|^2\rangle$",
        "message_total_l2_norm_mean": r"measured message norm $\langle\sum_e\|c_e\|\rangle$",
        "posterior_norm_mean": "posterior centered-logit norm",
        "tau_mean": r"mean dual variable $\tau$",
        "tau_shared": r"shared dual variable $\tau$",
        "shared_measured_cost": "shared-scope measured cost",
    }
    if x_key not in results:
        raise KeyError(f"x_key={x_key!r} not found. Available keys include: {sorted(results.keys())[:20]} ...")
    return np.asarray(results[x_key], dtype=np.float64), labels.get(x_key, x_key)


def _finite_mask_for_xy(x: np.ndarray, *ys: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    for y in ys:
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            mask &= np.isfinite(y_arr)
        elif y_arr.ndim >= 2:
            mask &= np.all(np.isfinite(y_arr.reshape(y_arr.shape[0], -1)), axis=1)
    return mask


def _safe_linthresh(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-8
    nonzero = np.abs(values[np.abs(values) > 0])
    if nonzero.size == 0:
        return 1e-8
    return max(float(np.min(nonzero)) / 2.0, 1e-8)


def _set_axis_scale(ax: plt.Axes, values: np.ndarray, *, axis: str, log: bool = True) -> None:
    """
    Apply log scale if all finite values are positive; otherwise apply symlog.
    Does nothing when log=False.
    """
    if not log:
        return
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    pos = finite[finite > 0]
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if pos.size == finite.size:
        setter("log")
    else:
        setter("symlog", linthresh=_safe_linthresh(finite))


def _apply_xy_scales(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, log_x: bool, log_y: bool) -> None:
    _set_axis_scale(ax, x, axis="x", log=log_x)
    _set_axis_scale(ax, y, axis="y", log=log_y)


def _level_labels(n_levels: int):
    return [fr"$\ell={ell}$" for ell in range(1, n_levels + 1)]


def plot_loss_error(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot test loss and test error versus a chosen x-axis."""
    results = _as_results(results_or_path)
    x, xlabel = _extract_x(results, x_key)
    loss = np.asarray(results["loss_mean"], dtype=np.float64)
    err = np.asarray(results["error_mean"], dtype=np.float64)
    mask = _finite_mask_for_xy(x, loss, err)
    x = x[mask]
    loss = loss[mask]
    err = err[mask]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(x, loss, marker="o")
    _apply_xy_scales(axs[0], x, loss, log_x=log_x, log_y=log_y)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("mean test cross-entropy")
    axs[0].set_title("Loss")
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].plot(x, err, marker="o")
    _apply_xy_scales(axs[1], x, err, log_x=log_x, log_y=log_y)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("mean top-1 error")
    axs[1].set_title("Error")
    axs[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_budget_diagnostics(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot measured message cost, posterior norm, and tau versus x."""
    results = _as_results(results_or_path)
    x, xlabel = _extract_x(results, x_key)
    series = [
        ("message_total_cost_mean", r"mean $\sum_e\|c_e\|^2$", "measured message cost"),
        ("posterior_norm_mean", "mean posterior centered-logit norm", "posterior norm"),
        ("tau_shared" if "tau_shared" in results else "tau_mean", r"$\tau$", "dual variable"),
        ("budget_hit_fraction", "budget hit fraction", "budget convergence"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, title) in zip(axs.flat, series):
        if key not in results:
            ax.axis("off")
            continue
        y = np.asarray(results[key], dtype=np.float64)
        mask = _finite_mask_for_xy(x, y)
        xx = x[mask]
        yy = y[mask]
        ax.plot(xx, yy, marker="o")
        _apply_xy_scales(ax, xx, yy, log_x=log_x, log_y=log_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_hierarchy_observables(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot level-wise hierarchy quantities versus x."""
    results = _as_results(results_or_path)
    x, xlabel = _extract_x(results, x_key)
    arrays = [
        ("A_mass_mean", r"mean mass on $A_\ell$", r"$P(A_\ell)$"),
        ("B_mass_mean", r"mean mass on $B_\ell$", r"$P(B_\ell)$"),
        ("margin_mean", r"mean margin $M_\ell$", r"$M_\ell$"),
        ("margin_pos_frac", r"fraction $M_\ell>0$", "fraction"),
        ("hier_acc", "hierarchical accuracy", r"$P(\arg\max q\in A_\ell)$"),
        ("valid_level_frac", "valid level fraction", "fraction"),
    ]

    n_levels = np.asarray(results["margin_mean"]).shape[1]
    labels = _level_labels(n_levels)
    fig, axs = plt.subplots(2, 3, figsize=(17, 8), sharex=False)

    for ax, (key, title, ylabel) in zip(axs.flat, arrays):
        if key not in results:
            ax.axis("off")
            continue
        arr = np.asarray(results[key], dtype=np.float64)
        mask = _finite_mask_for_xy(x, arr)
        xx = x[mask]
        aa = arr[mask]
        for ell in range(n_levels):
            ax.plot(xx, aa[:, ell], marker="o", label=labels[ell])
        _apply_xy_scales(ax, xx, aa, log_x=log_x, log_y=log_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    handles, leg_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=n_levels, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_loss_decomposition(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot level penalties log(1+exp(-M_l)), residual, and total loss."""
    results = _as_results(results_or_path)
    x, xlabel = _extract_x(results, x_key)
    penalties = np.asarray(results["level_penalty_mean"], dtype=np.float64)
    residual = np.asarray(results["residual_mean"], dtype=np.float64)
    loss = np.asarray(results["loss_mean"], dtype=np.float64)
    n_levels = penalties.shape[1]
    mask = _finite_mask_for_xy(x, penalties, residual, loss)
    x = x[mask]
    penalties = penalties[mask]
    residual = residual[mask]
    loss = loss[mask]

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for ell in range(n_levels):
        axs[0].plot(x, penalties[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _apply_xy_scales(axs[0], x, penalties, log_x=log_x, log_y=log_y)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(r"mean $\log(1+e^{-M_\ell})$")
    axs[0].set_title("Level penalties")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(frameon=False)

    axs[1].plot(x, residual, marker="o")
    _apply_xy_scales(axs[1], x, residual, log_x=log_x, log_y=log_y)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("mean residual")
    axs[1].set_title("Residual inside final set")
    axs[1].grid(True, which="both", alpha=0.3)

    axs[2].plot(x, loss, marker="o", label="actual loss")
    recon = residual + penalties.sum(axis=1)
    axs[2].plot(x, recon, marker="x", linestyle="--", label="residual + penalties")
    _apply_xy_scales(axs[2], x, np.column_stack([loss, recon]), log_x=log_x, log_y=log_y)
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel("loss")
    axs[2].set_title("Loss reconstruction")
    axs[2].grid(True, which="both", alpha=0.3)
    axs[2].legend(frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_peeled_loss_and_margin_fraction(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    loss_key: str = "level_penalty_mean",
    pos_key: str = "margin_pos_frac",
    levels: Optional[Sequence[int]] = None,
    figsize: Tuple[float, float] = (12, 4),
    title_prefix: str = "Global-budget BP",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot side by side:
      1. peeled test loss: <log(1 + exp(-M_l))>
      2. fraction of positive margins: Pr(M_l > 0)

    levels uses zero-based indices. If None, all levels are plotted.
    """
    results = _as_results(results_or_path)
    x, xlabel = _extract_x(results, x_key)
    peeled_loss = np.asarray(results[loss_key], dtype=float)
    margin_pos = np.asarray(results[pos_key], dtype=float)

    if peeled_loss.ndim != 2:
        raise ValueError(f"{loss_key} must have shape [n_lambda, n_levels].")
    if margin_pos.ndim != 2:
        raise ValueError(f"{pos_key} must have shape [n_lambda, n_levels].")

    mask = _finite_mask_for_xy(x, peeled_loss, margin_pos)
    x = x[mask]
    peeled_loss = peeled_loss[mask]
    margin_pos = margin_pos[mask]

    n_levels = peeled_loss.shape[1]
    if levels is None:
        levels = list(range(n_levels))

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for ell in levels:
        axs[0].plot(
            x,
            peeled_loss[:, ell],
            marker="o",
            linewidth=2,
            markersize=4,
            label=fr"$\ell={ell+1}$",
        )
    _apply_xy_scales(axs[0], x, peeled_loss[:, list(levels)], log_x=log_x, log_y=log_y)
    axs[0].set_title(f"{title_prefix}: peeled test loss")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(r"$\langle \log(1+e^{-M_\ell}) \rangle$")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(frameon=False)

    for ell in levels:
        axs[1].plot(
            x,
            margin_pos[:, ell],
            marker="o",
            linewidth=2,
            markersize=4,
            label=fr"$\ell={ell+1}$",
        )
    _apply_xy_scales(axs[1], x, margin_pos[:, list(levels)], log_x=log_x, log_y=log_y)
    axs[1].set_title(f"{title_prefix}: fraction of positive margins")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(r"$\Pr(M_\ell>0)$")
    # Only enforce linear-looking bounds if user disabled y log. With log/symlog,
    # these bounds can hide zero values or compress the meaningful transition.
    if not log_y:
        axs[1].set_ylim(-0.03, 1.03)
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend(frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_validity_by_position(
    results_or_path: Dict[str, Any] | str | Path,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    For next-token runs, plot which levels are actually valid as a function of target position.
    This is a heatmap, so log_x/log_y do not apply.
    """
    results = _as_results(results_or_path)
    if "valid_masks" not in results or "task_target_pos" not in results:
        raise KeyError("Need valid_masks and task_target_pos in results.")
    valid = np.asarray(results["valid_masks"], dtype=bool)  # [tasks, levels]
    pos = np.asarray(results["task_target_pos"], dtype=np.int64)
    levels = valid.shape[1]
    unique_pos = np.unique(pos)
    mat = np.zeros((levels, unique_pos.size), dtype=np.float64)
    for j, p in enumerate(unique_pos):
        sel = pos == p
        if np.any(sel):
            mat[:, j] = valid[sel].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(7, 0.5 * unique_pos.size), 4.5))
    im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(levels))
    ax.set_yticklabels([fr"$\ell={i}$" for i in range(1, levels + 1)])
    ax.set_xticks(np.arange(unique_pos.size))
    ax.set_xticklabels([str(int(p)) for p in unique_pos])
    ax.set_xlabel("target position i (zero-based)")
    ax.set_ylabel("hierarchical level")
    ax.set_title("Fraction of tasks where level is valid")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("valid fraction")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, ax


def plot_all(
    results_or_path: Dict[str, Any] | str | Path,
    out_prefix: Optional[str | Path] = None,
    x_key: str = "lambda_values",
    *,
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    """Generate the standard set of plots."""
    results = _as_results(results_or_path)
    prefix = None if out_prefix is None else Path(out_prefix)
    save = lambda suffix: None if prefix is None else Path(str(prefix) + suffix)

    plot_loss_error(results, x_key=x_key, save_path=save(f"_loss_error_vs_{x_key}.png"), log_x=log_x, log_y=log_y)
    plot_budget_diagnostics(results, x_key=x_key, save_path=save(f"_budget_diagnostics_vs_{x_key}.png"), log_x=log_x, log_y=log_y)
    plot_hierarchy_observables(results, x_key=x_key, save_path=save(f"_hierarchy_vs_{x_key}.png"), log_x=log_x, log_y=log_y)
    plot_loss_decomposition(results, x_key=x_key, save_path=save(f"_loss_decomposition_vs_{x_key}.png"), log_x=log_x, log_y=log_y)
    plot_peeled_loss_and_margin_fraction(results, x_key=x_key, save_path=save(f"_peeled_loss_margin_fraction_vs_{x_key}.png"), log_x=log_x, log_y=log_y)

    # Also plot against measured posterior norm if available and different from x_key.
    if x_key != "posterior_norm_mean" and "posterior_norm_mean" in results:
        plot_loss_error(results, x_key="posterior_norm_mean", save_path=save("_loss_error_vs_posterior_norm.png"), log_x=log_x, log_y=log_y)
        plot_hierarchy_observables(results, x_key="posterior_norm_mean", save_path=save("_hierarchy_vs_posterior_norm.png"), log_x=log_x, log_y=log_y)
        plot_peeled_loss_and_margin_fraction(results, x_key="posterior_norm_mean", save_path=save("_peeled_loss_margin_fraction_vs_posterior_norm.png"), log_x=log_x, log_y=log_y)

    params = results.get("params", {})
    if params.get("prediction_mode", None) == "next" or np.unique(np.asarray(results.get("task_target_pos", []))).size > 1:
        plot_validity_by_position(results, save_path=save("_validity_by_position.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot constrained BP results saved as .npz")
    parser.add_argument("--input", required=True, help="Path to .npz file produced by constrained_bp_last_next_torch.py")
    parser.add_argument("--out_prefix", default=None, help="Prefix for saved figures. If omitted, show only.")
    parser.add_argument("--x_key", default="lambda_values", help="x-axis key, e.g. lambda_values, posterior_norm_mean, message_total_cost_mean")
    parser.add_argument("--no_log_x", action="store_true", help="Disable log/symlog scaling on x axes.")
    parser.add_argument("--no_log_y", action="store_true", help="Disable log/symlog scaling on y axes.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    args = parser.parse_args()

    results = load_results(args.input)
    plot_all(
        results,
        out_prefix=args.out_prefix,
        x_key=args.x_key,
        log_x=not args.no_log_x,
        log_y=not args.no_log_y,
    )
    if args.show or args.out_prefix is None:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
