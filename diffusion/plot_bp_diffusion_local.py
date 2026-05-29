#!/usr/bin/env python3
"""Plot helpers for constrained_bp_diffusion_local.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str | Path) -> Dict[str, Any]:
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


def _extract_x(res: Dict[str, Any], x_key: str):
    labels = {
        "lambda_values": r"local message radius $\lambda$",
        "message_max_norm_mean": r"mean max message norm",
        "message_total_cost_mean": r"mean $\sum_e \|c_e\|_2^2$",
        "message_total_l2_norm_mean": r"mean $\sum_e \|c_e\|_2$",
        "posterior_norm_mean": r"posterior centered-logit norm",
    }
    if x_key not in res:
        raise KeyError(f"x_key={x_key!r} not found. Available keys: {sorted(res.keys())}")
    return np.asarray(res[x_key], dtype=float), labels.get(x_key, x_key)


def _finite_mask(x: np.ndarray, *ys: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    for y in ys:
        arr = np.asarray(y, dtype=float)
        if arr.ndim == 1:
            mask &= np.isfinite(arr)
        elif arr.ndim >= 2:
            mask &= np.all(np.isfinite(arr.reshape(arr.shape[0], -1)), axis=1)
    return mask


def _safe_linthresh(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    nonzero = np.abs(values[np.abs(values) > 0])
    if nonzero.size == 0:
        return 1e-8
    return max(float(np.min(nonzero)) / 2.0, 1e-8)


def _set_scale(ax, values, axis: str, log: bool):
    if not log:
        return
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if np.all(finite > 0):
        setter("log")
    else:
        setter("symlog", linthresh=_safe_linthresh(finite))


def plot_loss_error(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    *,
    log_x: bool = True,
    log_y_loss: bool = True,
    log_y_error: bool = False,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    res = _as_results(results_or_path)
    x, xlabel = _extract_x(res, x_key)
    loss = np.asarray(res["loss_mean"], dtype=float)
    err = np.asarray(res["error_mean"], dtype=float)
    mask = _finite_mask(x, loss, err)
    x, loss, err = x[mask], loss[mask], err[mask]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(x, loss, marker="o")
    _set_scale(axs[0], x, "x", log_x)
    _set_scale(axs[0], loss, "y", log_y_loss)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("masked-diffusion denoising CE")
    axs[0].set_title("BP diffusion test loss")
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].plot(x, err, marker="o")
    _set_scale(axs[1], x, "x", log_x)
    _set_scale(axs[1], err, "y", log_y_error)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("top-1 token error")
    axs[1].set_title("BP denoising error")
    axs[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_margin_negative_and_peeled_loss(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    *,
    log_x: bool = True,
    log_y_loss: bool = True,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    res = _as_results(results_or_path)
    x, xlabel = _extract_x(res, x_key)
    neg = np.asarray(res["margin_neg_frac"], dtype=float)
    peeled = np.asarray(res["level_penalty_mean"], dtype=float)
    mask = _finite_mask(x, neg, peeled)
    x, neg, peeled = x[mask], neg[mask], peeled[mask]
    L = neg.shape[1]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ell in range(L):
        axs[0].plot(x, neg[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _set_scale(axs[0], x, "x", log_x)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(r"$\Pr[M_\ell<0]$")
    axs[0].set_title("negative-margin probability")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend()

    for ell in range(L):
        axs[1].plot(x, peeled[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _set_scale(axs[1], x, "x", log_x)
    _set_scale(axs[1], peeled, "y", log_y_loss)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(r"$\langle\log(1+e^{-M_\ell})\rangle$")
    axs[1].set_title("level-wise peeled loss")
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def plot_loss_decomposition(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    *,
    log_x: bool = True,
    log_y: bool = True,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    res = _as_results(results_or_path)
    x, xlabel = _extract_x(res, x_key)
    loss = np.asarray(res["loss_mean"], dtype=float)
    residual = np.asarray(res["residual_mean"], dtype=float)
    peeled = np.asarray(res["level_penalty_all_mean"], dtype=float)
    recon = residual + np.sum(peeled, axis=1)
    mask = _finite_mask(x, loss, residual, peeled, recon)
    x, loss, residual, peeled, recon = x[mask], loss[mask], residual[mask], peeled[mask], recon[mask]
    L = peeled.shape[1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, loss, marker="o", linewidth=2.5, label="actual CE")
    ax.plot(x, recon, marker="x", linestyle="--", label="residual + levels")
    ax.plot(x, residual, marker="o", linestyle=":", label="residual")
    for ell in range(L):
        ax.plot(x, peeled[:, ell], marker="o", alpha=0.85, label=fr"level $\ell={ell+1}$")
    _set_scale(ax, x, "x", log_x)
    _set_scale(ax, np.concatenate([loss, recon, residual, peeled.reshape(-1)]), "y", log_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss contribution")
    ax.set_title("exact CE decomposition")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, ax


def plot_message_diagnostics(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "lambda_values",
    *,
    log_x: bool = True,
    log_y: bool = True,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    res = _as_results(results_or_path)
    x, xlabel = _extract_x(res, x_key)
    keys = [
        ("message_total_cost_mean", r"$\langle\sum_e\|c_e\|^2\rangle$", "total quadratic cost"),
        ("message_total_l2_norm_mean", r"$\langle\sum_e\|c_e\|\rangle$", "total L2 sum"),
        ("message_max_norm_mean", "mean max message norm", "max local norm"),
        ("message_clipped_fraction_mean", "clipped fraction", "fraction of clipped messages"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, title) in zip(axs.flat, keys):
        y = np.asarray(res[key], dtype=float)
        mask = _finite_mask(x, y)
        xx, yy = x[mask], y[mask]
        ax.plot(xx, yy, marker="o")
        _set_scale(ax, xx, "x", log_x)
        _set_scale(ax, yy, "y", log_y if key != "message_clipped_fraction_mean" else False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot constrained BP diffusion local results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument("--x_key", default="lambda_values")
    args = parser.parse_args()

    res = load_results(args.input)
    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = str(Path(args.input).with_suffix(""))
    plot_loss_error(res, x_key=args.x_key, save_path=f"{out_prefix}_loss_error.png")
    plot_margin_negative_and_peeled_loss(res, x_key=args.x_key, save_path=f"{out_prefix}_margins_peeled.png")
    plot_loss_decomposition(res, x_key=args.x_key, save_path=f"{out_prefix}_decomposition.png")
    plot_message_diagnostics(res, x_key=args.x_key, save_path=f"{out_prefix}_message_diag.png")
    print("Saved plots with prefix", out_prefix)


if __name__ == "__main__":
    main()
