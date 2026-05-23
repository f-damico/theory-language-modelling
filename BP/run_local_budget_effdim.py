#!/usr/bin/env python3
"""
Run constrained-message oracle belief propagation (EFFECTIVE-DIMENSION VERSION) on a sampled RHM instance and
sweep a range of local-message budgets lambdas.

This is the local-budget version: `lambda_values` are per-message centered-logit
L2 radii. Compared to the original local-budget runner, this version also
supports the repo's exact Zipf/layer dataset convention, including application
of the non-uniform rule probabilities on any selected layer.

In addition, the runner computes a lightweight logit-geometry diagnostic for
last-token prediction: R2 measures how much centered posterior-logit energy is
explained by the nested partitions {B_1,...,B_k,A_k}, while R_bar subtracts and
normalizes by the analytic shuffled-token baseline. No full logits are stored.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from BP.BeliefPropagation_local_budget import run_last_token_inference

EPS = 1e-12


def dec2base(values: np.ndarray, base: int, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((values.shape[0], length), dtype=np.int64)
    tmp = values.copy()
    for pos in range(length - 1, -1, -1):
        out[:, pos] = tmp % base
        tmp //= base
    return out


def _zipf_probabilities(m: int, zipf: float) -> np.ndarray:
    """Exact repo convention: p_r proportional to (r+1)^(-1-zipf) for r=0,...,m-1."""
    zipf_prob = np.ones(m, dtype=np.float64)
    for i in range(m):
        zipf_prob[i] = (i + 1) ** (-1.0 - float(zipf))
    zipf_prob /= np.sum(zipf_prob)
    return zipf_prob


def dec2base_torch(values: torch.Tensor, base: int, length: int) -> torch.Tensor:
    values = values.to(torch.int64).reshape(-1)
    out = torch.zeros((values.shape[0], length), dtype=torch.int64)
    tmp = values.clone()
    for pos in range(length - 1, -1, -1):
        out[:, pos] = tmp % base
        tmp = torch.div(tmp, base, rounding_mode='floor')
    return out


def sample_rules(v: int, n: int, m: int, s: int, L: int, seed: int = 42) -> List[np.ndarray]:
    """Exact repo tree sampler, returned as numpy arrays for the BP code."""
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))
    rules_torch: List[torch.Tensor] = []
    rules_torch.append(torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1))
    for _ in range(1, L):
        rules_torch.append(torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1))
    return [r.cpu().numpy().astype(np.int64, copy=False) for r in rules_torch]


def _rules_numpy_to_torch(rules: Sequence[np.ndarray]) -> List[torch.Tensor]:
    return [torch.from_numpy(np.asarray(r, dtype=np.int64)) for r in rules]


def sample_data_from_labels_torch(labels: torch.Tensor, rules_torch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact repo replacement=True, zipf=None path."""
    L = len(rules_torch)
    features = labels
    for l in range(L):
        chosen_rule = torch.randint(low=0, high=rules_torch[l].shape[1], size=features.shape)
        features = rules_torch[l][features, chosen_rule].flatten(start_dim=1)
    return features, labels


def sample_data_from_labels_prob_torch(
    labels: torch.Tensor,
    rules_torch: Sequence[torch.Tensor],
    layer: int,
    prob: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact repo replacement=True, zipf!=None path."""
    L = len(rules_torch)
    features = labels
    for l in range(L):
        if l == (layer - 1):
            chosen_rule = torch.multinomial(prob, features.numel(), replacement=True).reshape(features.shape)
        else:
            chosen_rule = torch.randint(low=0, high=rules_torch[l].shape[1], size=features.shape)
        features = rules_torch[l][features, chosen_rule].flatten(start_dim=1)
    return features, labels


def sample_data_from_indices_torch(
    samples: torch.Tensor,
    rules_torch: Sequence[torch.Tensor],
    n: int,
    m: int,
    s: int,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact repo without-replacement path, minus bonus outputs."""
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n

    high_level = samples.div(data_per_hl, rounding_mode='floor')
    low_level = samples % data_per_hl

    labels = high_level
    features = labels
    size = 1

    for l in range(L):
        choices = m**size
        data_per_hl = data_per_hl // choices
        high_level = low_level.div(data_per_hl, rounding_mode='floor')
        high_level = dec2base_torch(high_level, m, length=size).squeeze()
        features = rules_torch[l][features, high_level]
        features = features.flatten(start_dim=1)
        size *= s
        low_level = low_level % data_per_hl

    return features, labels


def build_rule_probabilities(
    rules: Sequence[np.ndarray],
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
) -> List[np.ndarray]:
    rule_probs: List[np.ndarray] = []
    for level_rules in rules:
        num_parents, m_here = level_rules.shape[:2]
        rule_probs.append(np.full((num_parents, m_here), 1.0 / m_here, dtype=np.float64))

    if zipf is not None:
        if layer is None:
            raise ValueError('zipf law requires layer of application')
        if not (1 <= int(layer) <= len(rules)):
            raise ValueError(f'layer must lie in [1, {len(rules)}], got {layer}')
        p = _zipf_probabilities(rules[int(layer) - 1].shape[1], zipf)
        rule_probs[int(layer) - 1][:] = p[None, :]
    return rule_probs


def build_train_test_dataset(
    num_features: int,
    num_classes: int,
    num_synonyms: int,
    tuple_size: int,
    num_layers: int,
    train_size: int,
    test_size: int,
    seed_rules: int,
    seed_sample: int,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build the dataset with the exact repo sampling convention.

    - If zipf is None and replacement=False, this matches the repo's default
      without-replacement branch.
    - If zipf is not None, the exact repo convention is to use replacement=True,
      torch.manual_seed(seed_sample), random labels from torch.randint, and
      torch.multinomial on the selected layer.

    `last_layer_powerlaw_a` is kept as a convenience alias to `zipf`; when used
    alone it defaults to `layer=1` (top/root expansion).
    """
    if zipf is None and last_layer_powerlaw_a is not None:
        zipf = float(last_layer_powerlaw_a)
        if layer is None:
            layer = 1

    if replacement is None:
        replacement = zipf is not None

    rules = sample_rules(
        v=num_features,
        n=num_classes,
        m=num_synonyms,
        s=tuple_size,
        L=num_layers,
        seed=seed_rules,
    )
    rule_probs = build_rule_probabilities(rules, zipf=zipf, layer=layer)
    rules_torch = _rules_numpy_to_torch(rules)

    max_data = num_classes * num_synonyms ** ((tuple_size**num_layers - 1) // (tuple_size - 1))
    if train_size < -1:
        raise ValueError('train_size must be greater than or equal to -1')

    if not replacement:
        if train_size == -1:
            samples = torch.arange(max_data, dtype=torch.int64)
        else:
            test_size = min(test_size, max_data - train_size)
            random.seed(seed_sample)
            samples = torch.tensor(random.sample(range(max_data), train_size + test_size), dtype=torch.int64)
        features_t, labels_t = sample_data_from_indices_torch(
            samples=samples,
            rules_torch=rules_torch,
            n=num_classes,
            m=num_synonyms,
            s=tuple_size,
            L=num_layers,
        )
    else:
        torch.manual_seed(seed_sample)
        if train_size == -1:
            total_size = max_data + test_size
        else:
            total_size = train_size + test_size
        labels = torch.randint(low=0, high=num_classes, size=(total_size,))
        if zipf is None:
            features_t, labels_t = sample_data_from_labels_torch(labels, rules_torch)
        else:
            if layer is None:
                raise ValueError('zipf law requires layer of application')
            prob = torch.from_numpy(_zipf_probabilities(num_synonyms, zipf))
            features_t, labels_t = sample_data_from_labels_prob_torch(labels, rules_torch, int(layer), prob)

    sequences = features_t.cpu().numpy().astype(np.int64, copy=False)
    labels = labels_t.cpu().numpy().astype(np.int64, copy=False)

    return {
        'rules': rules,
        'rule_probs': rule_probs,
        'train_sequences': sequences[:train_size],
        'test_sequences': sequences[train_size: train_size + test_size],
        'train_labels': labels[:train_size],
        'test_labels': labels[train_size: train_size + test_size],
        'max_data': max_data,
    }


def centered_logit_l2_norm(prob: np.ndarray, eps: float = EPS) -> float:
    logp = np.log(np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0))
    centered = logp - np.mean(logp)
    return float(np.linalg.norm(centered))


def centered_logits_from_prob(prob: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return centered logits from a probability vector, fixing the softmax gauge."""
    logp = np.log(np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0))
    return logp - np.mean(logp)


def _block_explained_energy_from_centered_logits(
    centered_logits: np.ndarray,
    groups: Sequence[np.ndarray],
    eps: float = 1e-14,
) -> Tuple[float, int]:
    """
    Fraction of centered-logit L2 energy explained by block means.

    Each group is a boolean mask over vocabulary coordinates. Empty groups are
    ignored. The returned integer is the number of non-empty blocks actually
    used.
    """
    z = np.asarray(centered_logits, dtype=np.float64).reshape(-1)
    denom = float(np.dot(z, z))
    if (not np.isfinite(denom)) or denom <= eps:
        return 0.0, 0

    explained = 0.0
    num_nonempty = 0
    for group in groups:
        mask = np.asarray(group, dtype=bool).reshape(-1)
        n_group = int(np.sum(mask))
        if n_group <= 0:
            continue
        mean_group = float(np.mean(z[mask]))
        explained += n_group * mean_group * mean_group
        num_nonempty += 1

    r2 = explained / denom
    # Numerical roundoff can push the projection ratio a tiny bit outside [0, 1].
    return float(np.clip(r2, 0.0, 1.0)), num_nonempty


def hierarchical_logit_geometry_scores(
    posterior: np.ndarray,
    A_masks_one: np.ndarray,
    B_masks_one: np.ndarray,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute R2 and R_bar for one posterior over the vocabulary.

    For level k, the partition is {B_1,...,B_k,A_k}. R2 is the fraction of
    centered-logit energy explained by the projection onto vectors that are
    constant on those blocks.

    The shuffled baseline is not estimated by Monte Carlo. For a centered vector
    of length q randomly permuted over K non-empty blocks, the expected explained
    fraction is exactly (K-1)/(q-1). This is much faster and deterministic.

    R_bar = (R2 - R2_shuffle) / (1 - R2_shuffle), so shuffled structure gives
    R_bar ~= 0 and perfect block-constant logits give R_bar = 1. Values below
    zero mean worse-than-shuffled alignment.
    """
    A_masks_one = np.asarray(A_masks_one, dtype=bool)
    B_masks_one = np.asarray(B_masks_one, dtype=bool)
    num_levels = int(A_masks_one.shape[0])
    q = int(A_masks_one.shape[1])

    z = centered_logits_from_prob(posterior, eps=eps)
    r2 = np.zeros(num_levels, dtype=np.float64)
    r2_shuffled = np.zeros(num_levels, dtype=np.float64)
    r_bar = np.zeros(num_levels, dtype=np.float64)

    for k in range(1, num_levels + 1):
        groups = [B_masks_one[ell] for ell in range(k)]
        groups.append(A_masks_one[k - 1])

        r2_k, num_blocks = _block_explained_energy_from_centered_logits(z, groups)
        if q > 1 and num_blocks > 0:
            null_k = float(num_blocks - 1) / float(q - 1)
        else:
            null_k = 0.0

        denom = 1.0 - null_k
        if denom > eps:
            r_bar_k = (r2_k - null_k) / denom
        else:
            r_bar_k = np.nan

        r2[k - 1] = r2_k
        r2_shuffled[k - 1] = null_k
        r_bar[k - 1] = r_bar_k

    return r2, r2_shuffled, r_bar



def _effective_dimensions_from_eigvals(eigvals: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Effective dimensions from the eigenvalues of the covariance of centered logits.

    Returns:
      - entropy effective dimension: exp(H(p)), p_a = eig_a / sum eig
      - participation-ratio dimension: (sum eig)^2 / sum eig^2
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    if (not np.isfinite(total)) or total <= eps:
        return 0.0, 0.0

    weights = eigvals / total
    weights_pos = weights[weights > eps]
    if weights_pos.size == 0:
        d_entropy = 0.0
    else:
        d_entropy = float(np.exp(-np.sum(weights_pos * np.log(weights_pos))))

    denom = float(np.sum(eigvals * eigvals))
    d_pr = float((total * total) / denom) if denom > eps else 0.0
    return d_entropy, d_pr


def _logit_cloud_effective_dimension_from_sums(
    sum_z: np.ndarray,
    sum_zz: np.ndarray,
    n: int,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute the effective dimension of the BP output-logit cloud without storing logits.

    During the sweep we accumulate, for one lambda,

        sum_z  = sum_mu z_mu
        sum_zz = sum_mu z_mu z_mu^T

    with z_mu = log p_mu - mean_j log p_mu_j.  From these two sufficient
    statistics we build the covariance across test inputs,

        Cov[z] = E[z z^T] - E[z] E[z]^T,

    and compute its effective rank.  Memory is O(V^2), not O(N_test V).
    """
    if n <= 0:
        raise ValueError("Cannot compute logit effective dimension with n <= 0.")

    mean_z = np.asarray(sum_z, dtype=np.float64) / float(n)
    second_moment = np.asarray(sum_zz, dtype=np.float64) / float(n)
    cov = second_moment - np.outer(mean_z, mean_z)
    cov = 0.5 * (cov + cov.T)  # numerical symmetrization

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    d_entropy, d_pr = _effective_dimensions_from_eigvals(eigvals, eps=eps)

    return {
        "logit_input_variance": float(np.trace(cov)),
        "logit_effdim_entropy": float(d_entropy),
        "logit_effdim_pr": float(d_pr),
        "logit_cov_eigvals": eigvals,
    }

def build_inverse_rule_maps(rules: Sequence[np.ndarray]) -> List[Dict[Tuple[int, ...], int]]:
    inverse_maps: List[Dict[Tuple[int, ...], int]] = []
    for level_rules in rules:
        inv: Dict[Tuple[int, ...], int] = {}
        for parent in range(level_rules.shape[0]):
            for r in range(level_rules.shape[1]):
                key = tuple(int(x) for x in level_rules[parent, r])
                inv[key] = parent
        inverse_maps.append(inv)
    return inverse_maps


def is_block_generable(block: np.ndarray, inverse_maps_slice: Sequence[Dict[Tuple[int, ...], int]], s: int) -> bool:
    current = np.asarray(block, dtype=np.int64).reshape(-1)
    for inv in reversed(inverse_maps_slice):
        if current.size % s != 0:
            return False
        grouped = current.reshape(-1, s)
        next_states = np.empty(grouped.shape[0], dtype=np.int64)
        for i, grp in enumerate(grouped):
            parent = inv.get(tuple(int(x) for x in grp))
            if parent is None:
                return False
            next_states[i] = parent
        current = next_states
    return current.size == 1


def candidate_set_for_level(
    xi: np.ndarray,
    level: int,
    inverse_maps: Sequence[Dict[Tuple[int, ...], int]],
    s: int,
    q: int,
    L: int,
) -> np.ndarray:
    if not (1 <= level <= L):
        raise ValueError(f"level must lie in [1, L], got {level}")
    block_size = s**level
    block = np.asarray(xi[-block_size:], dtype=np.int64).copy()
    rules_slice = inverse_maps[L - level : L]
    out = np.zeros(q, dtype=bool)
    for y in range(q):
        block[-1] = y
        out[y] = is_block_generable(block, rules_slice, s=s)
    return out


def precompute_hierarchy_masks(
    sequences: np.ndarray,
    rules: Sequence[np.ndarray],
    s: int,
    q: int,
    L: int,
) -> Dict[str, np.ndarray]:
    inverse_maps = build_inverse_rule_maps(rules)
    n = sequences.shape[0]
    A_masks = np.zeros((n, L, q), dtype=bool)
    B_masks = np.zeros((n, L, q), dtype=bool)
    all_vocab = np.ones(q, dtype=bool)

    for i, xi in enumerate(sequences):
        prev = all_vocab.copy()
        for ell in range(1, L + 1):
            A = candidate_set_for_level(xi, ell, inverse_maps=inverse_maps, s=s, q=q, L=L)
            B = prev & (~A)
            A_masks[i, ell - 1] = A
            B_masks[i, ell - 1] = B
            prev = A
    return {'A_masks': A_masks, 'B_masks': B_masks}


def _safe_log_ratio(num: float, den: float, eps: float = EPS) -> float:
    return float(np.log(max(num, eps)) - np.log(max(den, eps)))


def evaluate_bp_on_sequences(
    rules: Sequence[np.ndarray],
    sequences: np.ndarray,
    lambda_msg: float,
    num_features: int,
    num_classes: int,
    tuple_size: int,
    hierarchy_masks: Dict[str, np.ndarray],
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if len(sequences) == 0:
        raise ValueError('evaluate_bp_on_sequences received an empty sequence batch.')

    losses: List[float] = []
    errors: List[float] = []
    posterior_norms: List[float] = []

    A_masks = np.asarray(hierarchy_masks['A_masks'], dtype=bool)
    B_masks = np.asarray(hierarchy_masks['B_masks'], dtype=bool)
    num_levels = A_masks.shape[1]

    A_mass_sum = np.zeros(num_levels, dtype=np.float64)
    B_mass_sum = np.zeros(num_levels, dtype=np.float64)
    margin_sum = np.zeros(num_levels, dtype=np.float64)
    margin_pos_count = np.zeros(num_levels, dtype=np.float64)
    hier_acc_count = np.zeros(num_levels, dtype=np.float64)
    penalty_sum = np.zeros(num_levels, dtype=np.float64)
    R2_sum = np.zeros(num_levels, dtype=np.float64)
    R2_shuffled_sum = np.zeros(num_levels, dtype=np.float64)
    R_bar_sum = np.zeros(num_levels, dtype=np.float64)

    # Online sufficient statistics for the geometry of the full BP logit function.
    # We do NOT store all logits. For each lambda we only keep sum z and sum z z^T.
    q_dim = int(num_features)
    logit_sum_z = np.zeros(q_dim, dtype=np.float64)
    logit_sum_zz = np.zeros((q_dim, q_dim), dtype=np.float64)
    logit_energy_sum = 0.0

    residual_sum = 0.0
    recon_abs_err_sum = 0.0

    for idx, xi in enumerate(sequences):
        posterior, pred, loss = run_last_token_inference(
            rules=rules,
            l=len(rules),
            q=num_features,
            xi=xi,
            s=tuple_size,
            num_classes=num_classes,
            lambda_msg=lambda_msg,
            rule_probs=rule_probs,
            root_prior=root_prior,
        )
        losses.append(loss)
        errors.append(float(pred != int(xi[-1])))
        posterior_norms.append(centered_logit_l2_norm(posterior))

        # Centered posterior logits z_mu = log p_mu - mean_j log p_mu_j.
        # These are accumulated online to compute the covariance over test inputs.
        z_centered = centered_logits_from_prob(posterior)
        logit_sum_z += z_centered
        logit_sum_zz += np.outer(z_centered, z_centered)
        logit_energy_sum += float(np.dot(z_centered, z_centered))

        R2_this, R2_shuffled_this, R_bar_this = hierarchical_logit_geometry_scores(
            posterior=posterior,
            A_masks_one=A_masks[idx],
            B_masks_one=B_masks[idx],
        )
        R2_sum += R2_this
        R2_shuffled_sum += R2_shuffled_this
        R_bar_sum += R_bar_this

        margins_this = []
        for ell in range(num_levels):
            A_mask = A_masks[idx, ell]
            B_mask = B_masks[idx, ell]
            pA = float(np.sum(posterior[A_mask]))
            pB = float(np.sum(posterior[B_mask]))
            margin = _safe_log_ratio(pA, pB)
            penalty = float(np.log1p(np.exp(-margin)))
            margins_this.append(margin)

            A_mass_sum[ell] += pA
            B_mass_sum[ell] += pB
            margin_sum[ell] += margin
            margin_pos_count[ell] += float(margin > 0.0)
            hier_acc_count[ell] += float(A_mask[pred])
            penalty_sum[ell] += penalty

        pAL = float(np.sum(posterior[A_masks[idx, -1]]))
        residual = _safe_log_ratio(pAL, float(posterior[int(xi[-1])]))
        reconstructed = residual + float(np.sum(np.log1p(np.exp(-np.asarray(margins_this, dtype=np.float64)))))
        residual_sum += residual
        recon_abs_err_sum += abs(reconstructed - loss)

    n = float(len(sequences))
    logit_cloud_stats = _logit_cloud_effective_dimension_from_sums(
        sum_z=logit_sum_z,
        sum_zz=logit_sum_zz,
        n=int(len(sequences)),
    )

    return {
        'lambda': float(lambda_msg),
        'loss_mean': float(np.mean(losses)),
        'loss_std': float(np.std(losses)),
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'posterior_norm_mean': float(np.mean(posterior_norms)),
        'posterior_norm_std': float(np.std(posterior_norms)),
        'num_samples': int(len(sequences)),
        'A_mass_mean': A_mass_sum / n,
        'B_mass_mean': B_mass_sum / n,
        'margin_mean': margin_sum / n,
        'margin_pos_frac': margin_pos_count / n,
        'hier_acc': hier_acc_count / n,
        'level_penalty_mean': penalty_sum / n,
        'R2_mean': R2_sum / n,
        'R2_shuffled_mean': R2_shuffled_sum / n,
        'R_bar_mean': R_bar_sum / n,
        'logit_energy_mean': float(logit_energy_sum / n),
        'logit_input_variance': logit_cloud_stats['logit_input_variance'],
        'logit_effdim_entropy': logit_cloud_stats['logit_effdim_entropy'],
        'logit_effdim_pr': logit_cloud_stats['logit_effdim_pr'],
        'logit_effdim_entropy_norm': float(logit_cloud_stats['logit_effdim_entropy'] / max(q_dim - 1, 1)),
        'logit_effdim_pr_norm': float(logit_cloud_stats['logit_effdim_pr'] / max(q_dim - 1, 1)),
        'logit_cov_eigvals': logit_cloud_stats['logit_cov_eigvals'],
        'residual_mean': float(residual_sum / n),
        'reconstructed_loss_abs_error_mean': float(recon_abs_err_sum / n),
    }


def simulate_bp_lambda_sweep(
    num_features: int = 32,
    num_classes: int = 32,
    num_synonyms: int = 8,
    tuple_size: int = 2,
    num_layers: int = 3,
    train_size: int = 32768,
    test_size: int = 2048,
    lambda_values: Optional[Sequence[float]] = None,
    seed_rules: int = 0,
    seed_sample: int = 0,
    max_test_samples: Optional[int] = None,
    root_prior: Optional[np.ndarray] = None,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main notebook-friendly entry point.

    If `zipf` and `layer` are set, the dataset is sampled with the exact repo
    replacement=True convention using torch.manual_seed(seed_sample),
    torch.randint for labels, and torch.multinomial on the selected layer.
    The grammar tree itself is still exactly the repo tree for the same
    seed_rules.
    """
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-2, 2, 25)))
    lambda_values = np.asarray(lambda_values, dtype=np.float64)

    data = build_train_test_dataset(
        num_features=num_features,
        num_classes=num_classes,
        num_synonyms=num_synonyms,
        tuple_size=tuple_size,
        num_layers=num_layers,
        train_size=train_size,
        test_size=test_size,
        seed_rules=seed_rules,
        seed_sample=seed_sample,
        zipf=zipf,
        layer=layer,
        replacement=replacement,
        last_layer_powerlaw_a=last_layer_powerlaw_a,
    )

    test_sequences = np.asarray(data['test_sequences'], dtype=np.int64)
    if max_test_samples is not None:
        test_sequences = test_sequences[:max_test_samples]
    if test_sequences.shape[0] == 0:
        raise ValueError('No test sequences are available. Reduce train_size or increase the finite dataset size.')

    hierarchy_masks = precompute_hierarchy_masks(
        sequences=test_sequences,
        rules=data['rules'],
        s=tuple_size,
        q=num_features,
        L=num_layers,
    )

    sweep: List[Dict[str, Any]] = []
    for lam in tqdm(lambda_values):
        sweep.append(evaluate_bp_on_sequences(
            rules=data['rules'],
            sequences=test_sequences,
            lambda_msg=float(lam),
            num_features=num_features,
            num_classes=num_classes,
            tuple_size=tuple_size,
            hierarchy_masks=hierarchy_masks,
            rule_probs=data['rule_probs'],
            root_prior=root_prior,
        ))

    results = {
        'params': {
            'num_features': int(num_features),
            'num_classes': int(num_classes),
            'num_synonyms': int(num_synonyms),
            'tuple_size': int(tuple_size),
            'num_layers': int(num_layers),
            'train_size': int(train_size),
            'test_size': int(test_size),
            'seed_rules': int(seed_rules),
            'seed_sample': int(seed_sample),
            'zipf': None if zipf is None else float(zipf),
            'layer': None if layer is None else int(layer),
            'replacement': None if replacement is None else bool(replacement),
            'last_layer_powerlaw_a': None if last_layer_powerlaw_a is None else float(last_layer_powerlaw_a),
        },
        'lambda_values': lambda_values,
        'loss_mean': np.array([r['loss_mean'] for r in sweep], dtype=np.float64),
        'loss_std': np.array([r['loss_std'] for r in sweep], dtype=np.float64),
        'error_mean': np.array([r['error_mean'] for r in sweep], dtype=np.float64),
        'error_std': np.array([r['error_std'] for r in sweep], dtype=np.float64),
        'posterior_norm_mean': np.array([r['posterior_norm_mean'] for r in sweep], dtype=np.float64),
        'posterior_norm_std': np.array([r['posterior_norm_std'] for r in sweep], dtype=np.float64),
        'A_mass_mean': np.stack([r['A_mass_mean'] for r in sweep], axis=0),
        'B_mass_mean': np.stack([r['B_mass_mean'] for r in sweep], axis=0),
        'margin_mean': np.stack([r['margin_mean'] for r in sweep], axis=0),
        'margin_pos_frac': np.stack([r['margin_pos_frac'] for r in sweep], axis=0),
        'hier_acc': np.stack([r['hier_acc'] for r in sweep], axis=0),
        'level_penalty_mean': np.stack([r['level_penalty_mean'] for r in sweep], axis=0),
        'R2_mean': np.stack([r['R2_mean'] for r in sweep], axis=0),
        'R2_shuffled_mean': np.stack([r['R2_shuffled_mean'] for r in sweep], axis=0),
        'R_bar_mean': np.stack([r['R_bar_mean'] for r in sweep], axis=0),
        'R2_delta_mean': np.diff(
            np.concatenate([
                np.zeros((len(sweep), 1), dtype=np.float64),
                np.stack([r['R2_mean'] for r in sweep], axis=0),
            ], axis=1),
            axis=1,
        ),
        'R_bar_delta_mean': np.diff(
            np.concatenate([
                np.zeros((len(sweep), 1), dtype=np.float64),
                np.stack([r['R_bar_mean'] for r in sweep], axis=0),
            ], axis=1),
            axis=1,
        ),
        'logit_energy_mean': np.array([r['logit_energy_mean'] for r in sweep], dtype=np.float64),
        'logit_input_variance': np.array([r['logit_input_variance'] for r in sweep], dtype=np.float64),
        'logit_effdim_entropy': np.array([r['logit_effdim_entropy'] for r in sweep], dtype=np.float64),
        'logit_effdim_pr': np.array([r['logit_effdim_pr'] for r in sweep], dtype=np.float64),
        'logit_effdim_entropy_norm': np.array([r['logit_effdim_entropy_norm'] for r in sweep], dtype=np.float64),
        'logit_effdim_pr_norm': np.array([r['logit_effdim_pr_norm'] for r in sweep], dtype=np.float64),
        'logit_cov_eigvals': np.stack([r['logit_cov_eigvals'] for r in sweep], axis=0),
        'residual_mean': np.array([r['residual_mean'] for r in sweep], dtype=np.float64),
        'reconstructed_loss_abs_error_mean': np.array([r['reconstructed_loss_abs_error_mean'] for r in sweep], dtype=np.float64),
        'raw_per_lambda': sweep,
        'rules': data['rules'],
        'rule_probs': data['rule_probs'],
        'train_sequences': data['train_sequences'],
        'test_sequences': test_sequences,
        'A_masks': hierarchy_masks['A_masks'],
        'B_masks': hierarchy_masks['B_masks'],
        'note': (
            'lambda is the maximal L2 norm of the centered log-message carried by every internal BP message. '
            'If zipf and layer are set, the dataset is sampled with the exact repo replacement=True convention '
            'using torch.manual_seed(seed_sample), torch.randint for labels, and torch.multinomial on the '
            'selected layer. The tree itself is exactly the repo tree for the same seed_rules. Hierarchy '
            'observables are computed from the final BP posterior and the exact RHM-compatible sets A_l, B_l. '
            'R2 measures the fraction of centered posterior-logit energy explained by the nested block partition '
            '{B_1,...,B_l,A_l}; R_bar=(R2-R2_shuffled)/(1-R2_shuffled) uses the analytic shuffled baseline. '
            'The logit effective-dimension diagnostics are computed online from sum_z and sum_zz over test inputs, '
            'so full logits are not saved.'
        ),
    }
    return results


def plot_loss_vs_lambda(results: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(results['lambda_values'], dtype=np.float64)
    ax.plot(x, results['loss_mean'], marker='o')
    pos = x[x > 0]
    if pos.size > 0:
        ax.set_xscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
        ax.set_yscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
    ax.set_xlabel('message budget $\lambda$')
    ax.set_ylabel('test loss')
    ax.set_title('Local-budget BP: loss')
    ax.grid(True, which='both', alpha=0.3)
    return ax


def plot_error_vs_lambda(results: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(results['lambda_values'], dtype=np.float64)
    ax.plot(x, results['error_mean'], marker='o')
    pos = x[x > 0]
    if pos.size > 0:
        ax.set_xscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
        ax.set_yscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
    ax.set_xlabel('message budget $\lambda$')
    ax.set_ylabel('test error')
    ax.set_title('Local-budget BP: error')
    ax.grid(True, which='both', alpha=0.3)
    return ax


def plot_both(results: Dict[str, Any], save_prefix: Optional[Path] = None) -> Tuple[plt.Figure, np.ndarray]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plot_loss_vs_lambda(results, ax=axs[0])
    plot_error_vs_lambda(results, ax=axs[1])
    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(str(save_prefix) + '_loss_error_vs_lambda.png', dpi=160, bbox_inches='tight')
    return fig, axs


def plot_hierarchy_observables(
    results: Dict[str, Any],
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    x = np.asarray(results['lambda_values'], dtype=np.float64)
    A_mass = np.asarray(results['A_mass_mean'], dtype=np.float64)
    margin = np.asarray(results['margin_mean'], dtype=np.float64)
    margin_pos = np.asarray(results['margin_pos_frac'], dtype=np.float64)
    hier_acc = np.asarray(results['hier_acc'], dtype=np.float64)
    n_levels = A_mass.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        (A_mass, 'Mean mass on $A_\ell$', 'mass on $A_\ell$'),
        (margin, 'Mean level margin $M_\ell$', '$M_\ell$'),
        (margin_pos, 'Fraction with $M_\ell>0$', 'fraction'),
        (hier_acc, 'Hierarchical accuracy', 'Pr[argmax q \in A_l]'),
    ]

    pos = x[x > 0]
    for ax, (arr, title, ylabel) in zip(axs.flat, metrics):
        for ell in range(n_levels):
            ax.plot(x, arr[:, ell], marker='o', label=fr'$\ell={ell+1}$')
        if pos.size > 0:
            ax.set_xscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.3)

    axs[1, 0].set_xlabel('message budget $\lambda$')
    axs[1, 1].set_xlabel('message budget $\lambda$')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=n_levels, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_prefix is not None:
        fig.savefig(str(save_prefix) + '_hierarchy_observables_vs_lambda.png', dpi=160, bbox_inches='tight')
    return fig, axs


def plot_logit_geometry_observables(
    results: Dict[str, Any],
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the hierarchical block-geometry diagnostics R2 and R_bar."""
    x = np.asarray(results['lambda_values'], dtype=np.float64)
    R2 = np.asarray(results['R2_mean'], dtype=np.float64)
    R_bar = np.asarray(results['R_bar_mean'], dtype=np.float64)
    dR2 = np.asarray(results['R2_delta_mean'], dtype=np.float64)
    dR_bar = np.asarray(results['R_bar_delta_mean'], dtype=np.float64)
    n_levels = R2.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        (R2, '$R^2$: block explained logit energy', '$R^2$'),
        (R_bar, r'$\bar R$: shuffled-normalized $R^2$', r'$\bar R$'),
        (dR2, r'$\Delta R^2$: incremental block energy', r'$\Delta R^2$'),
        (dR_bar, r'$\Delta \bar R$: incremental normalized energy', r'$\Delta \bar R$'),
    ]

    pos = x[x > 0]
    for ax, (arr, title, ylabel) in zip(axs.flat, metrics):
        for ell in range(n_levels):
            ax.plot(x, arr[:, ell], marker='o', label=fr'$\ell={ell+1}$')
        if pos.size > 0:
            ax.set_xscale('symlog', linthresh=max(float(np.min(pos)) / 2.0, 1e-6))
        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.3)

    axs[1, 0].set_xlabel('message budget $\lambda$')
    axs[1, 1].set_xlabel('message budget $\lambda$')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=n_levels, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_prefix is not None:
        fig.savefig(str(save_prefix) + '_logit_geometry_vs_lambda.png', dpi=160, bbox_inches='tight')
    return fig, axs



def plot_logit_effective_dimension(
    results: Dict[str, Any],
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot effective dimension diagnostics of the centered BP logit cloud."""
    required = [
        'logit_energy_mean',
        'logit_input_variance',
        'logit_effdim_entropy',
        'logit_effdim_pr',
        'logit_effdim_entropy_norm',
        'logit_effdim_pr_norm',
    ]
    missing = [k for k in required if k not in results]
    if missing:
        raise KeyError(
            f"Missing effective-dimension keys {missing}. Re-run simulate_bp_lambda_sweep "
            "with the modified run_local_budget.py."
        )

    x = np.asarray(results['lambda_values'], dtype=np.float64)
    pos = x[x > 0]
    linthresh = max(float(np.min(pos)) / 2.0, 1e-6) if pos.size > 0 else 1e-6

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(x, results['logit_energy_mean'], marker='o', label=r'$E\|\tilde z\|^2$')
    axs[0].plot(x, results['logit_input_variance'], marker='s', label=r'$\mathrm{Tr}\,\mathrm{Cov}(\tilde z)$')
    axs[0].set_title('Logit energy / input variance')
    axs[0].set_ylabel('energy')
    axs[0].legend()

    axs[1].plot(x, results['logit_effdim_entropy'], marker='o', label='entropy eff. dim.')
    axs[1].plot(x, results['logit_effdim_pr'], marker='s', label='PR eff. dim.')
    axs[1].set_title('Effective logit dimension')
    axs[1].set_ylabel('dimension')
    axs[1].legend()

    axs[2].plot(x, results['logit_effdim_entropy_norm'], marker='o', label='entropy / (V-1)')
    axs[2].plot(x, results['logit_effdim_pr_norm'], marker='s', label='PR / (V-1)')
    axs[2].set_title('Normalized effective dimension')
    axs[2].set_ylabel('normalized dimension')
    axs[2].legend()

    for ax in axs:
        ax.set_xscale('symlog', linthresh=linthresh)
        ax.set_xlabel('message budget $\lambda$')
        ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(str(save_prefix) + '_logit_effective_dimension_vs_lambda.png', dpi=160, bbox_inches='tight')
    return fig, axs

def _parse_lambda_values(args: argparse.Namespace) -> np.ndarray:
    if args.lambda_values is not None:
        vals = [float(x) for x in args.lambda_values.split(',') if x.strip()]
        return np.array(vals, dtype=np.float64)
    vals = np.logspace(args.lambda_log10_min, args.lambda_log10_max, args.lambda_num)
    if args.include_zero:
        vals = np.concatenate(([0.0], vals))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description='Constrained-message BP lambda sweep for the RHM (exact repo Zipf support)')
    parser.add_argument('--num_features', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=32)
    parser.add_argument('--num_synonyms', type=int, default=8)
    parser.add_argument('--tuple_size', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=2048)
    parser.add_argument('--seed_rules', type=int, default=0)
    parser.add_argument('--seed_sample', type=int, default=0)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--zipf', type=float, default=None,
                        help='Repo convention: p_r proportional to (r+1)^(-1-zipf) on the selected layer.')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer index in the repo convention, starting at 1 for the top/root expansion.')
    parser.add_argument('--replacement', action='store_true',
                        help='Force replacement=True dataset sampling. By default this is enabled automatically when zipf is set.')
    parser.add_argument('--lambda_values', type=str, default=None,
                        help='Comma-separated lambdas. If omitted, use a logspace grid.')
    parser.add_argument('--lambda_log10_min', type=float, default=-2.0)
    parser.add_argument('--lambda_log10_max', type=float, default=2.0)
    parser.add_argument('--lambda_num', type=int, default=25)
    parser.add_argument('--include_zero', action='store_true')
    parser.add_argument('--out_prefix', type=str, default='/mnt/data/constrained_bp_rhm_zipf_exact')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    lambda_values = _parse_lambda_values(args)
    results = simulate_bp_lambda_sweep(
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_synonyms=args.num_synonyms,
        tuple_size=args.tuple_size,
        num_layers=args.num_layers,
        train_size=args.train_size,
        test_size=args.test_size,
        lambda_values=lambda_values,
        seed_rules=args.seed_rules,
        seed_sample=args.seed_sample,
        max_test_samples=args.max_test_samples,
        zipf=args.zipf,
        layer=args.layer,
        replacement=args.replacement if args.replacement else None,
    )

    out_prefix = Path(args.out_prefix)
    np.savez(
        str(out_prefix) + '.npz',
        lambda_values=results['lambda_values'],
        loss_mean=results['loss_mean'],
        loss_std=results['loss_std'],
        error_mean=results['error_mean'],
        error_std=results['error_std'],
        posterior_norm_mean=results['posterior_norm_mean'],
        posterior_norm_std=results['posterior_norm_std'],
        A_mass_mean=results['A_mass_mean'],
        B_mass_mean=results['B_mass_mean'],
        margin_mean=results['margin_mean'],
        margin_pos_frac=results['margin_pos_frac'],
        hier_acc=results['hier_acc'],
        level_penalty_mean=results['level_penalty_mean'],
        R2_mean=results['R2_mean'],
        R2_shuffled_mean=results['R2_shuffled_mean'],
        R_bar_mean=results['R_bar_mean'],
        R2_delta_mean=results['R2_delta_mean'],
        R_bar_delta_mean=results['R_bar_delta_mean'],
        logit_energy_mean=results['logit_energy_mean'],
        logit_input_variance=results['logit_input_variance'],
        logit_effdim_entropy=results['logit_effdim_entropy'],
        logit_effdim_pr=results['logit_effdim_pr'],
        logit_effdim_entropy_norm=results['logit_effdim_entropy_norm'],
        logit_effdim_pr_norm=results['logit_effdim_pr_norm'],
        logit_cov_eigvals=results['logit_cov_eigvals'],
        residual_mean=results['residual_mean'],
        reconstructed_loss_abs_error_mean=results['reconstructed_loss_abs_error_mean'],
        params_json=json.dumps(results['params']),
        note=results['note'],
    )

    print('Saved data to', str(out_prefix) + '.npz')
    print(results['note'])
    print('lambda values:', results['lambda_values'])
    print('loss mean:', results['loss_mean'])
    print('error mean:', results['error_mean'])
    print('margin mean:\n', results['margin_mean'])
    print('margin positive fraction:\n', results['margin_pos_frac'])
    print('hierarchical accuracy:\n', results['hier_acc'])
    print('R2 mean:\n', results['R2_mean'])
    print('R_bar mean:\n', results['R_bar_mean'])
    print('logit entropy effective dimension:', results['logit_effdim_entropy'])
    print('logit PR effective dimension:', results['logit_effdim_pr'])
    print('mean abs reconstruction error of grouped loss decomposition:', results['reconstructed_loss_abs_error_mean'])

    if not args.no_plots:
        plot_both(results, save_prefix=out_prefix)
        plot_hierarchy_observables(results, save_prefix=out_prefix)
        plot_logit_geometry_observables(results, save_prefix=out_prefix)
        plot_logit_effective_dimension(results, save_prefix=out_prefix)
        print('Saved plot to', str(out_prefix) + '_loss_error_vs_lambda.png')
        print('Saved plot to', str(out_prefix) + '_hierarchy_observables_vs_lambda.png')
        print('Saved plot to', str(out_prefix) + '_logit_geometry_vs_lambda.png')
        print('Saved plot to', str(out_prefix) + '_logit_effective_dimension_vs_lambda.png')


if __name__ == '__main__':
    main()
