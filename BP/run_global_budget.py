#!/usr/bin/env python3
"""
Run oracle BP on a sampled RHM instance with a *global* message budget and
sweep a range of total budgets.

This file keeps the same overall analysis pipeline as the original lambda-sweep
script:
- build one fixed RHM instance,
- evaluate BP next-token inference on the test set for a grid of controls,
- compute test loss / test error,
- compute the same four hierarchy observables from the final posterior and the
  exact RHM-compatible sets A_l, B_l,
- plot everything either against the target total budget or against a measured
  norm-like observable.

Interpretation of lambda
------------------------
Here `lambda_values` denote *target total message-cost budgets*:

    lambda_total = sum_e ||c_e||_2^2,

where c_e is the centered log-message of an internal BP message.

Internally, the BP code solves for a shared dual variable tau so that the
realized total cost approximately matches the requested lambda_total.  The final
posterior is then used to compute the same curves as before.
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

from BP.BeliefPropagation_global_budget import run_last_token_inference  # repo-style


EPS = 1e-12



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
        sample_ids = samples.cpu().numpy().astype(np.int64, copy=False)
    else:
        torch.manual_seed(seed_sample)
        if train_size == -1:
            labels = torch.randint(low=0, high=num_classes, size=(max_data + test_size,))
        else:
            labels = torch.randint(low=0, high=num_classes, size=(train_size + test_size,))
        if zipf is None:
            features_t, labels_t = sample_data_from_labels_torch(labels, rules_torch)
        else:
            if layer is None:
                raise ValueError('zipf law requires layer of application')
            prob = torch.from_numpy(_zipf_probabilities(num_synonyms, zipf))
            features_t, labels_t = sample_data_from_labels_prob_torch(labels, rules_torch, layer, prob)
        sample_ids = None

    sequences = features_t.cpu().numpy().astype(np.int64, copy=False)
    labels = labels_t.cpu().numpy().astype(np.int64, copy=False)

    return {
        'rules': rules,
        'rule_probs': rule_probs,
        'sample_ids': sample_ids,
        'train_sequences': sequences[:train_size],
        'test_sequences': sequences[train_size : train_size + test_size],
        'train_labels': labels[:train_size],
        'test_labels': labels[train_size : train_size + test_size],
        'max_data': max_data,
    }


def centered_logit_l2_norm(prob: np.ndarray, eps: float = EPS) -> float:
    logp = np.log(np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0))
    centered = logp - np.mean(logp)
    return float(np.linalg.norm(centered))


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
    return {"A_masks": A_masks, "B_masks": B_masks}


def _safe_log_ratio(num: float, den: float, eps: float = EPS) -> float:
    return float(np.log(max(num, eps)) - np.log(max(den, eps)))


def evaluate_bp_on_sequences(
    rules: Sequence[np.ndarray],
    sequences: np.ndarray,
    lambda_total: float,
    num_features: int,
    num_classes: int,
    tuple_size: int,
    hierarchy_masks: Dict[str, np.ndarray],
    root_prior: Optional[np.ndarray] = None,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    tau_guesses: Optional[np.ndarray] = None,
    budget_tol_rel: float = 5e-4,
    max_bisect_iter: int = 30,
) -> Dict[str, Any]:
    if len(sequences) == 0:
        raise ValueError('evaluate_bp_on_sequences received an empty sequence batch.')

    losses: List[float] = []
    errors: List[float] = []
    posterior_norms: List[float] = []
    total_costs: List[float] = []
    total_l2_norms: List[float] = []
    taus: List[float] = []
    converged: List[float] = []
    tau_last_per_sequence: List[float] = []

    A_masks = np.asarray(hierarchy_masks['A_masks'], dtype=bool)
    B_masks = np.asarray(hierarchy_masks['B_masks'], dtype=bool)
    num_levels = A_masks.shape[1]

    A_mass_sum = np.zeros(num_levels, dtype=np.float64)
    B_mass_sum = np.zeros(num_levels, dtype=np.float64)
    margin_sum = np.zeros(num_levels, dtype=np.float64)
    margin_pos_count = np.zeros(num_levels, dtype=np.float64)
    hier_acc_count = np.zeros(num_levels, dtype=np.float64)
    penalty_sum = np.zeros(num_levels, dtype=np.float64)
    residual_sum = 0.0
    recon_abs_err_sum = 0.0

    for idx, xi in enumerate(sequences):
        tau_guess = None if tau_guesses is None else float(tau_guesses[idx])
        posterior, pred, loss, bp_stats = run_last_token_inference(
            rules=rules,
            l=len(rules),
            q=num_features,
            xi=xi,
            s=tuple_size,
            num_classes=num_classes,
            lambda_total=lambda_total,
            rule_probs=rule_probs,
            root_prior=root_prior,
            tau_guess=tau_guess,
            budget_tol_rel=budget_tol_rel,
            max_bisect_iter=max_bisect_iter,
        )
        losses.append(loss)
        errors.append(float(pred != int(xi[-1])))
        posterior_norms.append(centered_logit_l2_norm(posterior))
        total_costs.append(float(bp_stats['total_cost']))
        total_l2_norms.append(float(bp_stats['total_l2_norm']))
        taus.append(float(bp_stats['tau']))
        tau_last_per_sequence.append(float(bp_stats['tau']))
        converged.append(float(bool(bp_stats['converged_to_budget'])))

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
    return {
        'lambda': float(lambda_total),
        'loss_mean': float(np.mean(losses)),
        'loss_std': float(np.std(losses)),
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'posterior_norm_mean': float(np.mean(posterior_norms)),
        'posterior_norm_std': float(np.std(posterior_norms)),
        'message_total_cost_mean': float(np.mean(total_costs)),
        'message_total_cost_std': float(np.std(total_costs)),
        'message_total_l2_norm_mean': float(np.mean(total_l2_norms)),
        'message_total_l2_norm_std': float(np.std(total_l2_norms)),
        'tau_mean': float(np.mean(taus)),
        'tau_std': float(np.std(taus)),
        'budget_hit_fraction': float(np.mean(converged)),
        'num_samples': int(len(sequences)),
        'A_mass_mean': A_mass_sum / n,
        'B_mass_mean': B_mass_sum / n,
        'margin_mean': margin_sum / n,
        'margin_pos_frac': margin_pos_count / n,
        'hier_acc': hier_acc_count / n,
        'level_penalty_mean': penalty_sum / n,
        'residual_mean': float(residual_sum / n),
        'reconstructed_loss_abs_error_mean': float(recon_abs_err_sum / n),
        'tau_last_per_sequence': np.asarray(tau_last_per_sequence, dtype=np.float64),
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
    warm_start_tau: bool = True,
    budget_tol_rel: float = 5e-4,
    max_bisect_iter: int = 30,
) -> Dict[str, Any]:
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
    tau_guess_vec: Optional[np.ndarray] = None
    for lam in tqdm(lambda_values):
        out = evaluate_bp_on_sequences(
            rules=data['rules'],
            sequences=test_sequences,
            lambda_total=float(lam),
            num_features=num_features,
            num_classes=num_classes,
            tuple_size=tuple_size,
            hierarchy_masks=hierarchy_masks,
            root_prior=root_prior,
            rule_probs=data['rule_probs'],
            tau_guesses=tau_guess_vec if warm_start_tau else None,
            budget_tol_rel=budget_tol_rel,
            max_bisect_iter=max_bisect_iter,
        )
        tau_guess_vec = np.asarray(out.pop('tau_last_per_sequence'), dtype=np.float64) if warm_start_tau else None
        sweep.append(out)

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
            'warm_start_tau': bool(warm_start_tau),
            'budget_tol_rel': float(budget_tol_rel),
            'max_bisect_iter': int(max_bisect_iter),
        },
        'lambda_values': lambda_values,
        'loss_mean': np.array([r['loss_mean'] for r in sweep], dtype=np.float64),
        'loss_std': np.array([r['loss_std'] for r in sweep], dtype=np.float64),
        'error_mean': np.array([r['error_mean'] for r in sweep], dtype=np.float64),
        'error_std': np.array([r['error_std'] for r in sweep], dtype=np.float64),
        'posterior_norm_mean': np.array([r['posterior_norm_mean'] for r in sweep], dtype=np.float64),
        'posterior_norm_std': np.array([r['posterior_norm_std'] for r in sweep], dtype=np.float64),
        'message_total_cost_mean': np.array([r['message_total_cost_mean'] for r in sweep], dtype=np.float64),
        'message_total_cost_std': np.array([r['message_total_cost_std'] for r in sweep], dtype=np.float64),
        'message_total_l2_norm_mean': np.array([r['message_total_l2_norm_mean'] for r in sweep], dtype=np.float64),
        'message_total_l2_norm_std': np.array([r['message_total_l2_norm_std'] for r in sweep], dtype=np.float64),
        'tau_mean': np.array([r['tau_mean'] for r in sweep], dtype=np.float64),
        'tau_std': np.array([r['tau_std'] for r in sweep], dtype=np.float64),
        'budget_hit_fraction': np.array([r['budget_hit_fraction'] for r in sweep], dtype=np.float64),
        'A_mass_mean': np.stack([r['A_mass_mean'] for r in sweep], axis=0),
        'B_mass_mean': np.stack([r['B_mass_mean'] for r in sweep], axis=0),
        'margin_mean': np.stack([r['margin_mean'] for r in sweep], axis=0),
        'margin_pos_frac': np.stack([r['margin_pos_frac'] for r in sweep], axis=0),
        'hier_acc': np.stack([r['hier_acc'] for r in sweep], axis=0),
        'level_penalty_mean': np.stack([r['level_penalty_mean'] for r in sweep], axis=0),
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
            'lambda is a target global quadratic budget on internal centered log-messages. '
            'For each sequence, BP solves for a shared dual variable tau so that the realized total cost '
            'sum_e ||c_e||_2^2 approximately matches lambda. If zipf and layer are set, the dataset is sampled '
            'with the exact repo replacement=True convention using torch.manual_seed(seed_sample), torch.randint '
            'for labels, and torch.multinomial on the selected layer. The tree itself is exactly the repo tree '
            'for the same seed_rules. Hierarchy observables are computed from the final BP posterior and the exact '
            'RHM-compatible sets A_l, B_l.'
        ),
    }
    return results


def _extract_x(results: Dict[str, Any], x_key: str) -> Tuple[np.ndarray, str]:
    x = np.asarray(results[x_key], dtype=np.float64)
    labels = {
        "lambda_values": "target total budget $\\lambda$",
        "message_total_cost_mean": "measured total message cost $\\sum_e ||c_e||_2^2$",
        "message_total_l2_norm_mean": "measured total message norm $\\sum_e ||c_e||_2$",
        "posterior_norm_mean": "measured posterior centered-logit norm",
        "tau_mean": "dual variable $\\tau$",
    }
    return x, labels.get(x_key, x_key)


def _set_x_axis(ax: plt.Axes, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    if pos.size == x.size:
        ax.set_xscale("log")
    elif pos.size > 0:
        linthresh = max(float(np.min(pos)) / 2.0, 1e-6)
        ax.set_xscale("symlog", linthresh=linthresh)


def plot_loss(results: Dict[str, Any], x_key: str = "lambda_values", ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x, xlabel = _extract_x(results, x_key)
    ax.plot(x, results["loss_mean"], marker="o")
    _set_x_axis(ax, x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test loss")
    ax.set_title("Global-budget BP: loss")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_error(results: Dict[str, Any], x_key: str = "lambda_values", ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x, xlabel = _extract_x(results, x_key)
    ax.plot(x, results["error_mean"], marker="o")
    _set_x_axis(ax, x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test error")
    ax.set_title("Global-budget BP: error")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_both(
    results: Dict[str, Any],
    x_key: str = "lambda_values",
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plot_loss(results, x_key=x_key, ax=axs[0])
    plot_error(results, x_key=x_key, ax=axs[1])
    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(str(save_prefix) + f"_loss_error_vs_{x_key}.png", dpi=160, bbox_inches="tight")
    return fig, axs


def plot_hierarchy_observables(
    results: Dict[str, Any],
    x_key: str = "lambda_values",
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    x, xlabel = _extract_x(results, x_key)
    A_mass = np.asarray(results["A_mass_mean"], dtype=np.float64)
    margin = np.asarray(results["margin_mean"], dtype=np.float64)
    margin_pos = np.asarray(results["margin_pos_frac"], dtype=np.float64)
    hier_acc = np.asarray(results["hier_acc"], dtype=np.float64)
    n_levels = A_mass.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metrics = [
        (A_mass, "Mean mass on $A_\\ell$", "mass on $A_\\ell$"),
        (margin, "Mean level margin $M_\\ell$", "$M_\\ell$"),
        (margin_pos, "Fraction with $M_\\ell>0$", "fraction"),
        (hier_acc, "Hierarchical accuracy", "$\\Pr[\\arg\\max q \\in A_\\ell]$"),
    ]

    for ax, (arr, title, ylabel) in zip(axs.flat, metrics):
        for ell in range(n_levels):
            ax.plot(x, arr[:, ell], marker="o", label=fr"$\ell={ell+1}$")
        _set_x_axis(ax, x)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    axs[1, 0].set_xlabel(xlabel)
    axs[1, 1].set_xlabel(xlabel)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_levels, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_prefix is not None:
        fig.savefig(str(save_prefix) + f"_hierarchy_observables_vs_{x_key}.png", dpi=160, bbox_inches="tight")
    return fig, axs




def plot_loss_decomposition(
    results: Dict[str, Any],
    x_key: str = "lambda_values",
    save_prefix: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    x, xlabel = _extract_x(results, x_key)
    penalties = np.asarray(results["level_penalty_mean"], dtype=np.float64)
    residual = np.asarray(results["residual_mean"], dtype=np.float64)
    n_levels = penalties.shape[1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for ell in range(n_levels):
        axs[0].plot(x, penalties[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _set_x_axis(axs[0], x)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(r"mean $\log(1+e^{-M_\ell})$")
    axs[0].set_title("Level penalties in the loss decomposition")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(frameon=False)

    axs[1].plot(x, residual, marker="o")
    _set_x_axis(axs[1], x)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("mean residual")
    axs[1].set_title(r"Residual $-\log(p_y / P_{A_L})$")
    axs[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(str(save_prefix) + f"_loss_decomposition_vs_{x_key}.png", dpi=160, bbox_inches="tight")
    return fig, axs


def plot_top_rule_probabilities(results: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    rule_probs = results.get("rule_probs", None)
    if rule_probs is None:
        raise ValueError("results does not contain rule_probs")
    top_probs = np.asarray(rule_probs[0][0], dtype=np.float64)
    ax.plot(np.arange(1, top_probs.size + 1), top_probs, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rule rank at top layer")
    ax.set_ylabel("probability")
    ax.set_title("Top-layer rule probability profile")
    ax.grid(True, which="both", alpha=0.3)
    return ax

def _parse_lambda_values(args: argparse.Namespace) -> np.ndarray:
    if args.lambda_values is not None:
        vals = [float(x) for x in args.lambda_values.split(",") if x.strip()]
        return np.array(vals, dtype=np.float64)
    vals = np.logspace(args.lambda_log10_min, args.lambda_log10_max, args.lambda_num)
    if args.include_zero:
        vals = np.concatenate(([0.0], vals))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Global-budget BP sweep for the RHM")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=2048)
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument('--zipf', type=float, default=None,
                        help='Repo convention: p_r proportional to (r+1)^(-1-zipf) on the selected layer.')
    parser.add_argument('--layer', type=int, default=None,
                        help='Repo convention: layer index in {1,...,L} where the Zipf law is applied.')
    parser.add_argument('--replacement', action='store_true',
                        help='Use the repo replacement=True dataset branch explicitly.')
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None,
                        help="If set, only the top/root production-rule probabilities use p_r proportional to r^{-(1+a)}.")
    parser.add_argument("--lambda_values", type=str, default=None,
                        help="Comma-separated target total budgets. If omitted, use a logspace grid.")
    parser.add_argument("--lambda_log10_min", type=float, default=-2.0)
    parser.add_argument("--lambda_log10_max", type=float, default=2.0)
    parser.add_argument("--lambda_num", type=int, default=25)
    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/global_budget_bp_rhm")
    parser.add_argument("--no_plots", action="store_true")
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
        replacement=True if args.replacement else None,
        last_layer_powerlaw_a=args.last_layer_powerlaw_a,
    )

    out_prefix = Path(args.out_prefix)
    np.savez(
        str(out_prefix) + ".npz",
        lambda_values=results["lambda_values"],
        loss_mean=results["loss_mean"],
        loss_std=results["loss_std"],
        error_mean=results["error_mean"],
        error_std=results["error_std"],
        posterior_norm_mean=results["posterior_norm_mean"],
        posterior_norm_std=results["posterior_norm_std"],
        message_total_cost_mean=results["message_total_cost_mean"],
        message_total_cost_std=results["message_total_cost_std"],
        message_total_l2_norm_mean=results["message_total_l2_norm_mean"],
        message_total_l2_norm_std=results["message_total_l2_norm_std"],
        tau_mean=results["tau_mean"],
        tau_std=results["tau_std"],
        budget_hit_fraction=results["budget_hit_fraction"],
        A_mass_mean=results["A_mass_mean"],
        B_mass_mean=results["B_mass_mean"],
        margin_mean=results["margin_mean"],
        margin_pos_frac=results["margin_pos_frac"],
        hier_acc=results["hier_acc"],
        level_penalty_mean=results["level_penalty_mean"],
        residual_mean=results["residual_mean"],
        reconstructed_loss_abs_error_mean=results["reconstructed_loss_abs_error_mean"],
        params_json=json.dumps(results["params"]),
        top_rule_probs=np.asarray(results["rule_probs"][0], dtype=np.float64),
        note=results["note"],
    )

    print("Saved data to", str(out_prefix) + ".npz")
    print(results["note"])
    print("lambda values:", results["lambda_values"])
    print("measured total cost mean:", results["message_total_cost_mean"])
    print("posterior norm mean:", results["posterior_norm_mean"])
    print("tau mean:", results["tau_mean"])
    print("loss mean:", results["loss_mean"])
    print("error mean:", results["error_mean"])
    print("margin mean:\n", results["margin_mean"])
    print("margin positive fraction:\n", results["margin_pos_frac"])
    print("hierarchical accuracy:\n", results["hier_acc"])
    print("mean abs reconstruction error of grouped loss decomposition:", results["reconstructed_loss_abs_error_mean"])

    if not args.no_plots:
        plot_both(results, x_key="lambda_values", save_prefix=out_prefix)
        plot_hierarchy_observables(results, x_key="lambda_values", save_prefix=out_prefix)
        plot_both(results, x_key="posterior_norm_mean", save_prefix=out_prefix)
        plot_hierarchy_observables(results, x_key="posterior_norm_mean", save_prefix=out_prefix)
        plot_loss_decomposition(results, x_key="lambda_values", save_prefix=out_prefix)
        plot_loss_decomposition(results, x_key="posterior_norm_mean", save_prefix=out_prefix)
        print("Saved plot to", str(out_prefix) + "_loss_error_vs_lambda_values.png")
        print("Saved plot to", str(out_prefix) + "_hierarchy_observables_vs_lambda_values.png")
        print("Saved plot to", str(out_prefix) + "_loss_error_vs_posterior_norm_mean.png")
        print("Saved plot to", str(out_prefix) + "_hierarchy_observables_vs_posterior_norm_mean.png")
        print("Saved plot to", str(out_prefix) + "_loss_decomposition_vs_lambda_values.png")
        print("Saved plot to", str(out_prefix) + "_loss_decomposition_vs_posterior_norm_mean.png")


if __name__ == "__main__":
    main()
