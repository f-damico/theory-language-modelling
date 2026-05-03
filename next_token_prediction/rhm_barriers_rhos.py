#!/usr/bin/env python3
"""
Compute exact RHM-compatible candidate sets/barriers and analytic rho values
for autoregressive next-token prediction on datasets generated exactly as in the
`theory-language-modelling` repo.

Main guarantees
---------------
1) The rule instance is sampled with the same Python `random.sample` logic and
   seed as the repo's `sample_rules`.
2) The dataset indices are sampled with the same Python `random.sample` logic
   and seed as the repo's `RandomHierarchyModel(..., seed_sample=...)` default
   without-replacement branch.
3) Leaves/tokens and top labels are generated with the same integer arithmetic
   as the repo's `sample_data_from_rules`.

This file can be:
- imported in a notebook and called through `compute_rhm_observables(...)`
- run from CLI / PBS and save a pickle file with all outputs.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt

import numpy as np


ArrayInt = np.ndarray


@dataclass(frozen=True)
class RHMParams:
    num_features: int  # vocabulary size v at leaves and hidden non-root levels
    num_classes: int   # root vocabulary size n
    num_synonyms: int  # multiplicity m
    tuple_size: int    # branching factor s
    num_layers: int    # tree depth L
    seed_rules: int
    seed_sample: int
    train_size: int
    test_size: int

    @property
    def sequence_length(self) -> int:
        return self.tuple_size ** self.num_layers

    @property
    def max_data(self) -> int:
        s = self.tuple_size
        L = self.num_layers
        return self.num_classes * self.num_synonyms ** ((s**L - 1) // (s - 1))


def dec2base_np(values: np.ndarray, base: int, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((values.shape[0], length), dtype=np.int64)
    tmp = values.copy()
    for pos in range(length - 1, -1, -1):
        out[:, pos] = tmp % base
        tmp //= base
    return out


def sample_rules_exact(params: RHMParams) -> List[np.ndarray]:
    """Exact replica of repo `sample_rules` using Python's `random.sample`."""
    v = params.num_features
    n = params.num_classes
    m = params.num_synonyms
    s = params.tuple_size
    L = params.num_layers

    random.seed(params.seed_rules)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules: List[np.ndarray] = []
    rules.append(np.array(random.sample(tuples, n * m), dtype=np.int64).reshape(n, m, s))
    for _ in range(1, L):
        rules.append(np.array(random.sample(tuples, v * m), dtype=np.int64).reshape(v, m, s))
    return rules


def sample_indices_exact(params: RHMParams) -> np.ndarray:
    """Exact replica of repo sample-index generation for the default branch."""
    max_data = params.max_data
    if params.train_size == -1:
        return np.arange(max_data, dtype=np.int64)

    test_size = min(params.test_size, max_data - params.train_size)
    random.seed(params.seed_sample)
    return np.array(
        random.sample(range(max_data), params.train_size + test_size),
        dtype=np.int64,
    )


def sample_data_from_indices_exact(
    samples: np.ndarray,
    rules: Sequence[np.ndarray],
    params: RHMParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact numpy replica of repo `sample_data_from_rules` without one-hot encoding."""
    n = params.num_classes
    m = params.num_synonyms
    s = params.tuple_size
    L = params.num_layers

    max_data = params.max_data
    data_per_hl = max_data // n

    high_level = samples // data_per_hl
    low_level = samples % data_per_hl

    labels = high_level.astype(np.int64, copy=True)
    features = labels.reshape(-1, 1)
    size = 1

    for l in range(L):
        choices = m ** size
        data_per_hl = data_per_hl // choices

        high_level = low_level // data_per_hl
        high_level = dec2base_np(high_level, m, length=size)

        next_features = np.empty((features.shape[0], size, s), dtype=np.int64)
        for b in range(features.shape[0]):
            for pos in range(size):
                parent_symbol = features[b, pos]
                rule_index = high_level[b, pos]
                next_features[b, pos, :] = rules[l][parent_symbol, rule_index, :]
        features = next_features.reshape(features.shape[0], size * s)
        size *= s
        low_level = low_level % data_per_hl

    return features.astype(np.int64), labels.astype(np.int64)


def generate_repo_exact_dataset(params: RHMParams) -> Dict[str, np.ndarray]:
    rules = sample_rules_exact(params)
    sample_indices = sample_indices_exact(params)
    features, labels = sample_data_from_indices_exact(sample_indices, rules, params)

    train_size = params.train_size if params.train_size != -1 else features.shape[0]
    test_size = min(params.test_size, features.shape[0] - train_size)

    return {
        "rules": rules,
        "sample_indices": sample_indices,
        "features_all": features,
        "labels_all": labels,
        "train_features": features[:train_size].copy(),
        "train_labels": labels[:train_size].copy(),
        "test_features": features[train_size:train_size + test_size].copy(),
        "test_labels": labels[train_size:train_size + test_size].copy(),
    }


def level_plateau_correlation(params: RHMParams, level: int) -> float:
    """Analytic RMS plateau from the paper for LCA height `level` (1..L)."""
    if level < 1 or level > params.num_layers:
        raise ValueError(f"level must lie in [1, {params.num_layers}], got {level}")
    v = float(params.num_features)
    m = float(params.num_synonyms)
    s = int(params.tuple_size)
    return math.sqrt((1.0 - m / (v ** (s - 1))) / (v**3 * m ** (2 * level - 1)))


def shell_child_index(position_1based: int, level: int, s: int) -> int:
    """Child index r_{i,l} in {1,...,s} for position i inside its level-l block."""
    return 1 + (((position_1based - 1) % (s**level)) // (s ** (level - 1)))


def compute_rhos(params: RHMParams) -> Dict[str, np.ndarray]:
    """
    Compute position-dependent and level-averaged rho values.

    Returns arrays with shape:
    - rho_by_position_level: [d, L+1], with level 0 filled with 0.
    - n_shell_by_position_level: [d, L+1], shell sizes.
    - plateau_corr_by_level: [L+1], level 0 filled with 0.
    - rho_avg_by_level: [L+1], position-averaged RMS rho.
    """
    d = params.sequence_length
    L = params.num_layers
    s = params.tuple_size

    rho = np.zeros((d, L + 1), dtype=np.float64)
    n_shell = np.zeros((d, L + 1), dtype=np.int64)
    plateau = np.zeros(L + 1, dtype=np.float64)

    for level in range(1, L + 1):
        plateau[level] = level_plateau_correlation(params, level)
        child_block = s ** (level - 1)
        for pos0 in range(d):
            i = pos0 + 1
            r_il = shell_child_index(i, level, s)
            n_shell[pos0, level] = (r_il - 1) * child_block
            rho[pos0, level] = math.sqrt(float(n_shell[pos0, level])) * plateau[level]

    rho_avg_by_level = rho.mean(axis=0)
    return {
        "rho_by_position_level": rho,
        "n_shell_by_position_level": n_shell,
        "plateau_corr_by_level": plateau,
        "rho_avg_by_level": rho_avg_by_level,
    }


class CompatibilityComputer:
    """Exact candidate-set computer for one fixed RHM rule instance."""

    def __init__(self, params: RHMParams, rules: Sequence[np.ndarray]):
        self.params = params
        self.rules = [np.asarray(r, dtype=np.int64) for r in rules]
        self.v = params.num_features
        self.n = params.num_classes
        self.s = params.tuple_size
        self.L = params.num_layers
        self._token_universe = frozenset(range(self.v))

    def _block_start_0based(self, i0: int, level: int) -> int:
        block = self.s ** level
        return (i0 // block) * block

    def _build_pattern(self, sequence: np.ndarray, i0: int, level: int) -> Tuple[int, ...]:
        """
        Pattern over the level-l block containing position i0.
        Observed leaves are those strictly left of the target position.
        Unknown leaves are encoded as -1.
        """
        start = self._block_start_0based(i0, level)
        block = self.s ** level
        pattern = np.full(block, -1, dtype=np.int64)
        target_local = i0 - start
        for local in range(target_local):
            pattern[local] = int(sequence[start + local])
        return tuple(int(x) for x in pattern)

    @lru_cache(maxsize=None)
    def _any_completion_under_symbol(
        self,
        depth: int,
        rule_idx: int,
        pattern: Tuple[int, ...],
        root_symbol: int,
    ) -> bool:
        if depth == 0:
            obs = pattern[0]
            return obs == -1 or obs == root_symbol

        child_size = self.s ** (depth - 1)
        rules_here = self.rules[rule_idx][root_symbol]  # [m, s]
        for children in rules_here:
            ok = True
            for child_idx in range(self.s):
                start = child_idx * child_size
                child_pattern = pattern[start:start + child_size]
                child_symbol = int(children[child_idx])
                if not self._any_completion_under_symbol(depth - 1, rule_idx + 1, child_pattern, child_symbol):
                    ok = False
                    break
            if ok:
                return True
        return False

    @lru_cache(maxsize=None)
    def _target_tokens_under_symbol(
        self,
        depth: int,
        rule_idx: int,
        pattern: Tuple[int, ...],
        target_local_pos: int,
        root_symbol: int,
    ) -> frozenset[int]:
        if depth == 0:
            obs = pattern[0]
            if obs != -1 and obs != root_symbol:
                return frozenset()
            return frozenset((root_symbol,))

        child_size = self.s ** (depth - 1)
        target_child = target_local_pos // child_size
        target_child_local = target_local_pos % child_size

        out: set[int] = set()
        rules_here = self.rules[rule_idx][root_symbol]
        for children in rules_here:
            consistent = True
            for child_idx in range(self.s):
                if child_idx == target_child:
                    continue
                start = child_idx * child_size
                child_pattern = pattern[start:start + child_size]
                child_symbol = int(children[child_idx])
                if not self._any_completion_under_symbol(depth - 1, rule_idx + 1, child_pattern, child_symbol):
                    consistent = False
                    break
            if not consistent:
                continue

            start = target_child * child_size
            child_pattern = pattern[start:start + child_size]
            child_symbol = int(children[target_child])
            out.update(
                self._target_tokens_under_symbol(
                    depth - 1,
                    rule_idx + 1,
                    child_pattern,
                    target_child_local,
                    child_symbol,
                )
            )
        return frozenset(out)

    def compatible_token_set(self, sequence: np.ndarray, position_1based: int, level: int) -> frozenset[int]:
        """Exact A_{i,l} for one sequence, target position i, and level l."""
        i0 = position_1based - 1
        pattern = self._build_pattern(sequence, i0, level)
        start = self._block_start_0based(i0, level)
        target_local = i0 - start
        rule_idx = self.L - level
        if rule_idx < 0:
            raise ValueError(f"level {level} exceeds num_layers {self.L}")

        # Whole-tree root uses num_classes symbols; all other subtree roots use num_features.
        if level == self.L:
            root_domain = self.n
        else:
            root_domain = self.v

        out: set[int] = set()
        for root_symbol in range(root_domain):
            out.update(
                self._target_tokens_under_symbol(
                    level,
                    rule_idx,
                    pattern,
                    target_local,
                    root_symbol,
                )
            )
        return frozenset(out)

    def compute_A_sizes_for_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Return |A_{i,l}| with shape [d, L+1]."""
        d = self.params.sequence_length
        L = self.L
        A_sizes = np.zeros((d, L + 1), dtype=np.int32)
        for pos0 in range(d):
            i = pos0 + 1
            for level in range(L + 1):
                A_sizes[pos0, level] = len(self.compatible_token_set(sequence, i, level))
        return A_sizes

    def compute_barriers_from_A_sizes(self, A_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return b and B sizes from A sizes. Shapes [d, L+1]."""
        d, Lp1 = A_sizes.shape
        B_sizes = np.zeros((d, Lp1), dtype=np.int32)
        barriers = np.full((d, Lp1), np.nan, dtype=np.float64)
        for pos0 in range(d):
            for level in range(1, Lp1):
                B_sizes[pos0, level] = int(A_sizes[pos0, level - 1] - A_sizes[pos0, level])
                if A_sizes[pos0, level] > 0 and B_sizes[pos0, level] > 0:
                    barriers[pos0, level] = math.log(B_sizes[pos0, level] / A_sizes[pos0, level])
                else:
                    barriers[pos0, level] = np.nan
        return barriers, B_sizes


def compute_barriers_for_dataset(
    sequences: np.ndarray,
    compatibility: CompatibilityComputer,
) -> Dict[str, np.ndarray]:
    num_samples = sequences.shape[0]
    d = compatibility.params.sequence_length
    L = compatibility.params.num_layers
    A_sizes = np.zeros((num_samples, d, L + 1), dtype=np.int16)
    B_sizes = np.zeros((num_samples, d, L + 1), dtype=np.int16)
    barriers = np.full((num_samples, d, L + 1), np.nan, dtype=np.float64)

    for idx in range(num_samples):
        A_seq = compatibility.compute_A_sizes_for_sequence(sequences[idx])
        b_seq, B_seq = compatibility.compute_barriers_from_A_sizes(A_seq)
        A_sizes[idx] = A_seq
        B_sizes[idx] = B_seq
        barriers[idx] = b_seq

    valid = np.isfinite(barriers)
    count = valid.sum(axis=0)
    barrier_mean = np.divide(
        np.nansum(barriers, axis=0),
        count,
        out=np.full((d, L + 1), np.nan, dtype=np.float64),
        where=count > 0,
    )
    sq_count = count
    barrier_std = np.sqrt(
        np.divide(
            np.nansum((barriers - barrier_mean[None, :, :]) ** 2, axis=0),
            sq_count,
            out=np.full((d, L + 1), np.nan, dtype=np.float64),
            where=sq_count > 0,
        )
    )
    A_mean = A_sizes.mean(axis=0)

    return {
        "A_sizes": A_sizes,
        "B_sizes": B_sizes,
        "barriers": barriers,
        "barrier_mean": barrier_mean,
        "barrier_std": barrier_std,
        "A_size_mean": A_mean,
    }


def compute_rhm_observables(
    num_features: int = 32,
    num_classes: int = 32,
    num_synonyms: int = 8,
    tuple_size: int = 2,
    num_layers: int = 3,
    seed_rules: int = 0,
    seed_sample: int = 0,
    train_size: int = 32768,
    test_size: int = 32768,
    compute_train: bool = True,
    compute_test: bool = True,
) -> Dict[str, object]:
    """
    Main notebook-friendly entry point.

    Returns a dictionary containing:
    - params
    - exact repo-matched rules / sample indices / datasets
    - rho arrays
    - exact A/B/barrier arrays for train/test (if requested)
    """
    params = RHMParams(
        num_features=num_features,
        num_classes=num_classes,
        num_synonyms=num_synonyms,
        tuple_size=tuple_size,
        num_layers=num_layers,
        seed_rules=seed_rules,
        seed_sample=seed_sample,
        train_size=train_size,
        test_size=test_size,
    )

    dataset = generate_repo_exact_dataset(params)
    compatibility = CompatibilityComputer(params, dataset["rules"])
    rho_info = compute_rhos(params)

    out: Dict[str, object] = {
        "params": asdict(params),
        "rules": dataset["rules"],
        "sample_indices": dataset["sample_indices"],
        "train_features": dataset["train_features"],
        "train_labels": dataset["train_labels"],
        "test_features": dataset["test_features"],
        "test_labels": dataset["test_labels"],
    }
    out.update(rho_info)

    if compute_train and dataset["train_features"].size > 0:
        out["train_observables"] = compute_barriers_for_dataset(dataset["train_features"], compatibility)

    if compute_test and dataset["test_features"].size > 0:
        out["test_observables"] = compute_barriers_for_dataset(dataset["test_features"], compatibility)

    return out


def _summarize(result: Dict[str, object]) -> str:
    params = result["params"]
    lines = [
        "Computed exact repo-matched RHM observables:",
        f"  v={params['num_features']} n={params['num_classes']} m={params['num_synonyms']} s={params['tuple_size']} L={params['num_layers']}",
        f"  seed_rules={params['seed_rules']} seed_sample={params['seed_sample']}",
        f"  train_size={params['train_size']} test_size={params['test_size']}",
        f"  sequence_length={params['tuple_size'] ** params['num_layers']}",
    ]
    rho_avg = result["rho_avg_by_level"]
    lines.append("  rho_avg_by_level=" + np.array2string(rho_avg, precision=6))
    if "train_observables" in result:
        lines.append(
            "  train barrier_mean shape=" + str(result["train_observables"]["barrier_mean"].shape)
        )
    if "test_observables" in result:
        lines.append(
            "  test barrier_mean shape=" + str(result["test_observables"]["barrier_mean"].shape)
        )
    return "\n".join(lines)


def plot_predicted_test_curves_from_rhm_observables(
    rhm_obs: dict,
    norm_grid=None,
    n_norm=300,
    norm_min=None,
    norm_max=None,
    use_test=True,
    average_over_positions=True,
    show=True,
):
    """
    Plot predicted test loss and approximate test error versus the global logit norm N,
    under the simple assumption phi_{i,l}^mu = 1.

    Parameters
    ----------
    rhm_obs : dict
        Output of compute_rhm_observables(...) from rhm_barriers_rhos.py.
    norm_grid : array-like or None
        Optional explicit grid of N values.
    n_norm : int
        Number of points if norm_grid is not provided.
    norm_min, norm_max : float or None
        Optional manual bounds for the norm grid.
    use_test : bool
        If True, use rhm_obs["test_observables"], else use train_observables.
    average_over_positions : bool
        If True, compute mean over all (mu, i).
        If False, returns also position-resolved curves averaged only over samples.
    show : bool
        Whether to show the plots.

    Returns
    -------
    out : dict
        Contains:
        - "norm_grid"
        - "pred_test_loss"
        - "pred_test_error_proxy"
        - optionally position-resolved versions
    """

    key = "test_observables" if use_test else "train_observables"
    if key not in rhm_obs:
        raise KeyError(f"{key!r} not found in rhm_obs")

    obs = rhm_obs[key]

    # Shapes from rhm_barriers_rhos.py:
    # A_sizes        : [num_samples, d, L+1]
    # B_sizes        : [num_samples, d, L+1]
    # barriers       : [num_samples, d, L+1]
    # valid_levels   : [num_samples, d, L+1]  bool
    # rho_by_position_level : [d, L+1]
    A_sizes = np.asarray(obs["A_sizes"], dtype=np.float64)
    barriers = np.asarray(obs["barriers"], dtype=np.float64)
    if "valid_levels" in obs:
        valid_levels = np.asarray(obs["valid_levels"], dtype=bool)
    else:
        B_sizes = np.asarray(obs["B_sizes"], dtype=np.float64)
        valid_levels = (
            np.isfinite(np.asarray(obs["barriers"], dtype=np.float64))
            & (B_sizes > 0)
            & (A_sizes > 0)
        )
    rho_pos_level = np.asarray(rhm_obs["rho_by_position_level"], dtype=np.float64)

    num_samples, d, Lp1 = A_sizes.shape
    L = Lp1 - 1

    # Broadcast rho to [num_samples, d, L+1]
    rho = np.broadcast_to(rho_pos_level[None, :, :], (num_samples, d, Lp1))

    # Deepest surviving set size |A_{i, L_i}^mu|:
    # use the last level with valid_levels == True
    deepest_idx = np.zeros((num_samples, d), dtype=np.int64)
    for ell in range(1, L + 1):
        deepest_idx = np.where(valid_levels[:, :, ell], ell, deepest_idx)

    # Gather |A_{i,L_i}^mu|
    A_final = np.take_along_axis(A_sizes, deepest_idx[:, :, None], axis=2).squeeze(axis=2)
    A_final = np.maximum(A_final, 1.0)

    # Build norm grid
    if norm_grid is None:
        # Thresholds N* ~ b / rho for rho > 0
        rho_pos = rho[:, :, 1:]
        b_pos = barriers[:, :, 1:]
        valid_pos = valid_levels[:, :, 1:] & (rho_pos > 0)

        if np.any(valid_pos):
            thr = b_pos[valid_pos] / np.maximum(rho_pos[valid_pos], 1e-300)
            thr = thr[np.isfinite(thr)]
        else:
            thr = np.array([1.0])

        if thr.size == 0:
            thr = np.array([1.0])

        if norm_min is None:
            norm_min = max(1e-6, 0.1 * np.percentile(thr, 5))
        if norm_max is None:
            norm_max = 2.0 * np.percentile(thr, 95)

        if norm_max <= norm_min:
            norm_max = max(norm_min * 10.0, norm_min + 1.0)

        norm_grid = np.logspace(np.log10(norm_min), np.log10(norm_max), n_norm)
    else:
        norm_grid = np.asarray(norm_grid, dtype=np.float64)

    # Output arrays
    pred_loss = np.zeros_like(norm_grid, dtype=np.float64)
    pred_error_proxy = np.zeros_like(norm_grid, dtype=np.float64)

    if not average_over_positions:
        pred_loss_by_pos = np.zeros((d, norm_grid.size), dtype=np.float64)
        pred_error_proxy_by_pos = np.zeros((d, norm_grid.size), dtype=np.float64)

    # Helper: stable softplus(-x) = log(1 + exp(-x))
    def softplus_neg(x):
        return np.logaddexp(0.0, -x)

    # Loop over N
    for k, N in enumerate(norm_grid):
        M = N * rho - barriers  # [num_samples, d, L+1]

        # Peeling contribution only for valid levels ell >= 1
        peel = np.zeros_like(M)
        peel[:, :, 1:] = softplus_neg(M[:, :, 1:])
        peel = np.where(valid_levels, peel, 0.0)

        # Residual term: log |A_final|
        residual = np.log(A_final)

        # Position/sample loss
        loss_per_item = residual + np.sum(peel[:, :, 1:], axis=2)  # [num_samples, d]
        pred_loss[k] = loss_per_item.mean()

        # Approximate correct probability:
        # p_corr ≈ |A_final|^{-1} prod_l sigma(M_l)
        log_p_corr = -residual.copy()
        for ell in range(1, L + 1):
            # log sigma(M) = -softplus(-M)
            contrib = -softplus_neg(M[:, :, ell])
            contrib = np.where(valid_levels[:, :, ell], contrib, 0.0)
            log_p_corr += contrib

        p_corr = np.exp(np.clip(log_p_corr, -700, 50))
        pred_error_proxy[k] = 1.0 - p_corr.mean()

        if not average_over_positions:
            pred_loss_by_pos[:, k] = loss_per_item.mean(axis=0)
            pred_error_proxy_by_pos[:, k] = 1.0 - p_corr.mean(axis=0)

    # Plot
    if show:
        plt.figure(figsize=(6, 4))
        plt.plot(norm_grid, pred_loss, lw=2)
        plt.xscale("log")
        plt.xlabel("global logit norm $N$", fontsize=13)
        plt.ylabel("predicted test loss", fontsize=13)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(norm_grid, pred_error_proxy, lw=2)
        plt.xscale("log")
        plt.xlabel("global logit norm $N$", fontsize=13)
        plt.ylabel("predicted test error (proxy)", fontsize=13)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

    out = {
        "norm_grid": norm_grid,
        "pred_test_loss": pred_loss,
        "pred_test_error_proxy": pred_error_proxy,
    }

    if not average_over_positions:
        out["pred_test_loss_by_position"] = pred_loss_by_pos
        out["pred_test_error_proxy_by_position"] = pred_error_proxy_by_pos

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact RHM rho and barrier observables.")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=32768)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="Output pickle file.")
    return parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt


def plot_margin_learning_order(
    rhm_obs,
    split="test",
    norm_grid=None,
    n_norm=300,
    norm_min=None,
    norm_max=None,
    eps=1e-12,
    show=True,
):
    """
    Plot the predicted learning order of M_{i,l}^mu under

        M_{i,l}^mu(N) = N * rho_{i,l} - b_{i,l}^mu

    i.e. phi = 1 and one global logit norm N.

    Produces:
    1. fraction of solved margins M>0 versus N, one curve per level;
    2. threshold distribution N* = b/rho by level.

    Parameters
    ----------
    rhm_obs : dict
        Output of compute_rhm_observables(...).
    split : {"train", "test"}
        Which observables to use.
    norm_grid : array-like or None
        Optional explicit N grid.
    n_norm : int
        Number of grid points if norm_grid is None.
    norm_min, norm_max : float or None
        Manual N range.
    eps : float
        Numerical epsilon.
    show : bool
        Whether to display plots.

    Returns
    -------
    out : dict
        Contains thresholds, masks, norm_grid, and solved fractions.
    """

    obs_key = f"{split}_observables"
    if obs_key not in rhm_obs:
        raise KeyError(f"{obs_key!r} not found in rhm_obs.")

    obs = rhm_obs[obs_key]

    A = np.asarray(obs["A_sizes"], dtype=np.float64)
    B = np.asarray(obs["B_sizes"], dtype=np.float64)
    b = np.asarray(obs["barriers"], dtype=np.float64)

    rho_pos_level = np.asarray(rhm_obs["rho_by_position_level"], dtype=np.float64)

    num_samples, d, Lp1 = A.shape
    L = Lp1 - 1

    rho = np.broadcast_to(rho_pos_level[None, :, :], (num_samples, d, Lp1))

    # We ignore ell=0 because it is A_0 = V, not a peeling level.
    A1 = A[:, :, 1:]
    B1 = B[:, :, 1:]
    b1 = b[:, :, 1:]
    rho1 = rho[:, :, 1:]

    # Cases:
    # valid: nontrivial barrier and positive signal
    valid = (
        np.isfinite(b1)
        & (A1 > 0)
        & (B1 > 0)
        & (rho1 > eps)
    )

    # Already solved/trivial:
    # no excluded group, or negative barrier gives M(0) = -b > 0.
    trivial = (
        ((B1 <= 0) | np.isneginf(b1) | ((b1 < 0) & (rho1 > eps)))
        & (A1 > 0)
    )

    # Impossible under this simple rho-only model:
    # wrong group exists but rho is zero.
    never = (
        (B1 > 0)
        & (A1 > 0)
        & (rho1 <= eps)
    )

    # Thresholds N* = b/rho.
    # If b < 0, the margin is already positive at N=0, so threshold is 0.
    Nstar = np.full_like(b1, np.nan, dtype=np.float64)
    Nstar[valid] = np.maximum(b1[valid] / rho1[valid], 0.0)

    finite_thresholds = Nstar[np.isfinite(Nstar)]

    if norm_grid is None:
        if finite_thresholds.size > 0:
            positive_thr = finite_thresholds[finite_thresholds > 0]
            if positive_thr.size == 0:
                positive_thr = np.array([1.0])

            if norm_min is None:
                norm_min = max(1e-8, 0.2 * np.percentile(positive_thr, 5))
            if norm_max is None:
                norm_max = 5.0 * np.percentile(positive_thr, 95)
        else:
            if norm_min is None:
                norm_min = 1e-3
            if norm_max is None:
                norm_max = 1e3

        if norm_max <= norm_min:
            norm_max = norm_min * 10.0

        norm_grid = np.logspace(np.log10(norm_min), np.log10(norm_max), n_norm)
    else:
        norm_grid = np.asarray(norm_grid, dtype=np.float64)

    # Fraction solved by level.
    # A term is solved if it is trivial or if N >= Nstar.
    frac_solved = np.zeros((L, norm_grid.size), dtype=np.float64)
    frac_never = np.zeros(L, dtype=np.float64)
    frac_trivial = np.zeros(L, dtype=np.float64)
    counts_effective = np.zeros(L, dtype=np.int64)

    for ell0 in range(L):
        valid_l = valid[:, :, ell0]
        trivial_l = trivial[:, :, ell0]
        never_l = never[:, :, ell0]

        denominator_mask = valid_l | trivial_l | never_l
        denom = int(np.sum(denominator_mask))
        counts_effective[ell0] = denom

        if denom == 0:
            frac_solved[ell0, :] = np.nan
            frac_never[ell0] = np.nan
            frac_trivial[ell0] = np.nan
            continue

        frac_never[ell0] = np.sum(never_l) / denom
        frac_trivial[ell0] = np.sum(trivial_l) / denom

        Ns_l = Nstar[:, :, ell0]

        for k, N in enumerate(norm_grid):
            solved_l = trivial_l | (valid_l & (Ns_l <= N))
            frac_solved[ell0, k] = np.sum(solved_l) / denom

    if show:
        # 1. Solved-fraction curves
        plt.figure(figsize=(7, 4.5))
        for ell in range(1, L + 1):
            plt.plot(
                norm_grid,
                frac_solved[ell - 1],
                lw=2,
                label=fr"$\ell={ell}$",
            )

        plt.xscale("log")
        plt.ylim(-0.02, 1.02)
        plt.xlabel(r"global logit norm $N$")
        plt.ylabel(r"fraction with $M_{i,\ell}^{\mu}(N)>0$")
        plt.title("Predicted order of margin peeling")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. Threshold distributions
        data = []
        labels = []
        for ell in range(1, L + 1):
            vals = Nstar[:, :, ell - 1]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size > 0:
                data.append(np.log10(vals))
                labels.append(fr"$\ell={ell}$")

        if len(data) > 0:
            plt.figure(figsize=(7, 4.5))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.ylabel(r"$\log_{10} N^*_{i,\ell,\mu}$")
            plt.title(r"Distribution of learning thresholds $N^*=b/\rho$")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 3. Diagnostic fractions
        x = np.arange(1, L + 1)

        plt.figure(figsize=(7, 4.2))
        plt.plot(x, frac_trivial, marker="o", label="trivial / already solved")
        plt.plot(x, frac_never, marker="o", label="never learned if rho=0")
        plt.xlabel(r"level $\ell$")
        plt.ylabel("fraction of terms")
        plt.title("Degenerate terms by level")
        plt.xticks(x)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "norm_grid": norm_grid,
        "Nstar": Nstar,
        "valid": valid,
        "trivial": trivial,
        "never": never,
        "frac_solved": frac_solved,
        "frac_trivial": frac_trivial,
        "frac_never": frac_never,
        "counts_effective": counts_effective,
    }


def main() -> None:
    args = parse_args()
    result = compute_rhm_observables(
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_synonyms=args.num_synonyms,
        tuple_size=args.tuple_size,
        num_layers=args.num_layers,
        seed_rules=args.seed_rules,
        seed_sample=args.seed_sample,
        train_size=args.train_size,
        test_size=args.test_size,
        compute_train=not args.skip_train,
        compute_test=not args.skip_test,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(result, handle)
    print(_summarize(result))
    print(f"Saved output to {args.out}")


if __name__ == "__main__":
    main()
