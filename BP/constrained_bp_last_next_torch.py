#!/usr/bin/env python3
"""
Torch implementation of constrained-message oracle BP on the RHM for
last-token and next-token prediction.

What this file does
-------------------
Given RHM parameters and seeds, it:
  1) builds the sampled RHM rules with the same convention as the previous code,
  2) builds a train/test dataset with the same seed conventions,
  3) creates prediction tasks either for last-token prediction or next-token prediction,
  4) runs global-budget BP for a grid of lambda values,
  5) returns/saves loss, error, message costs, posterior norms, and hierarchy observables.

Two budget scopes are supported:
  - shared: one tau is fitted for the whole evaluation set/batch collection, so the
    mean total message cost per prediction is approximately lambda_total. This is
    the more transformer-like option.
  - per_inference: one tau is fitted independently for each prediction problem,
    reproducing the spirit of the older global-budget code.

The algorithm is still the global centered-log-message shrinkage version, not a
full variational optimizer of D_KL(p* || q) over messages.

Prediction modes
----------------
  - last: for every sequence, mask only the last token and predict it from all
    previous tokens.
  - next: for every selected position i, clamp the prefix x_<i, mask target i,
    leave future tokens x_>i unobserved, and predict x_i.

All BP computations are performed in PyTorch and can run on CPU or GPU via
--device cpu/cuda.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

EPS = 1e-12
MASK_VALUE_DEFAULT = None  # replaced by q at runtime


# -----------------------------------------------------------------------------
# Basic tensor utilities
# -----------------------------------------------------------------------------


def _as_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return dev


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-30) -> torch.Tensor:
    """Normalize nonnegative values along dim, with uniform fallback for bad rows."""
    total = x.sum(dim=dim, keepdim=True)
    bad = (~torch.isfinite(total)) | (total <= eps)
    if bad.any():
        x = x.clone()
        # set bad rows to ones along the normalized dimension
        expanded_bad = bad.expand_as(x)
        x[expanded_bad] = 1.0
        total = x.sum(dim=dim, keepdim=True)
    return x / total.clamp_min(eps)


def centered_log_probs(prob: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Centered log-message c = log p - mean(log p)."""
    logp = torch.log(prob.clamp_min(eps))
    return logp - logp.mean(dim=-1, keepdim=True)


def softmax_centered(c: torch.Tensor) -> torch.Tensor:
    """Softmax after removing the max for numerical stability."""
    return torch.softmax(c, dim=-1)


def centered_logit_l2_norm_torch(prob: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    c = centered_log_probs(prob, eps=eps)
    return torch.linalg.vector_norm(c, ord=2, dim=-1)


def dec2base_torch(values: torch.Tensor, base: int, length: int) -> torch.Tensor:
    values = values.to(torch.int64).reshape(-1)
    out = torch.zeros((values.shape[0], length), dtype=torch.int64, device=values.device)
    tmp = values.clone()
    for pos in range(length - 1, -1, -1):
        out[:, pos] = tmp % base
        tmp = torch.div(tmp, base, rounding_mode="floor")
    return out


# -----------------------------------------------------------------------------
# Exact repo-style RHM construction and sampling
# -----------------------------------------------------------------------------


def zipf_probabilities(m: int, zipf: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ranks = torch.arange(1, m + 1, dtype=dtype, device=device)
    p = ranks.pow(-1.0 - float(zipf))
    return p / p.sum()


def sample_rules(
    v: int,
    n: int,
    m: int,
    s: int,
    L: int,
    seed: int = 42,
    *,
    device: str | torch.device = "cpu",
) -> List[torch.Tensor]:
    """
    Exact previous-code convention:
      - level 0/root rules have n parents,
      - lower rules have v parents,
      - each level samples parent*m distinct s-tuples without replacement from V^s.
    """
    dev = _as_device(device)
    random.seed(int(seed))
    tuples = list(product(*[range(v) for _ in range(s)]))
    rules: List[torch.Tensor] = []
    root = torch.tensor(random.sample(tuples, n * m), dtype=torch.long, device=dev).reshape(n, m, s)
    rules.append(root)
    for _ in range(1, L):
        r = torch.tensor(random.sample(tuples, v * m), dtype=torch.long, device=dev).reshape(v, m, s)
        rules.append(r)
    return rules


def build_rule_probabilities(
    rules: Sequence[torch.Tensor],
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> List[torch.Tensor]:
    """Return rule-probability tables [num_parents, m] for every level."""
    out: List[torch.Tensor] = []
    for r in rules:
        num_parents, m = int(r.shape[0]), int(r.shape[1])
        out.append(torch.full((num_parents, m), 1.0 / m, dtype=dtype, device=r.device))

    if zipf is not None:
        if layer is None:
            raise ValueError("zipf requires a layer in {1,...,L}.")
        if not (1 <= int(layer) <= len(rules)):
            raise ValueError(f"layer must lie in [1,{len(rules)}], got {layer}.")
        idx = int(layer) - 1
        m = int(rules[idx].shape[1])
        p = zipf_probabilities(m, float(zipf), device=rules[idx].device, dtype=dtype)
        out[idx] = p[None, :].expand(rules[idx].shape[0], m).clone()
    return out


def sample_data_from_labels_torch(labels: torch.Tensor, rules: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact repo replacement=True, zipf=None path."""
    features = labels.to(torch.long).reshape(-1, 1)
    for level_rules in rules:
        B, width = features.shape
        gen = torch.Generator(device=features.device)
        # This function assumes the caller already set the global generator seed if desired.
        chosen_rule = torch.randint(0, int(level_rules.shape[1]), size=features.shape, device=features.device)
        features = level_rules[features, chosen_rule].flatten(start_dim=1)
    return features, labels


def sample_data_from_labels_prob_torch(
    labels: torch.Tensor,
    rules: Sequence[torch.Tensor],
    layer: int,
    prob: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact repo replacement=True, zipf!=None path."""
    features = labels.to(torch.long).reshape(-1, 1)
    for l, level_rules in enumerate(rules):
        if l == int(layer) - 1:
            chosen_rule = torch.multinomial(prob, features.numel(), replacement=True).reshape(features.shape)
        else:
            chosen_rule = torch.randint(0, int(level_rules.shape[1]), size=features.shape, device=features.device)
        features = level_rules[features, chosen_rule].flatten(start_dim=1)
    return features, labels


def sample_data_from_indices_torch(
    samples: torch.Tensor,
    rules: Sequence[torch.Tensor],
    n: int,
    m: int,
    s: int,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact previous-code without-replacement dataset branch."""
    samples = samples.to(torch.long)
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n

    high_level = torch.div(samples, data_per_hl, rounding_mode="floor")
    low_level = samples % data_per_hl

    labels = high_level
    features = labels.reshape(-1, 1)
    size = 1

    for l in range(L):
        choices = m**size
        data_per_hl = data_per_hl // choices
        high_level = torch.div(low_level, data_per_hl, rounding_mode="floor")
        high_level = dec2base_torch(high_level, m, length=size).reshape(features.shape)
        features = rules[l][features, high_level].flatten(start_dim=1)
        size *= s
        low_level = low_level % data_per_hl

    return features, labels


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
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Build rules and train/test sequences with the previous code's conventions."""
    dev = _as_device(device)
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
        device=dev,
    )
    rule_probs = build_rule_probabilities(rules, zipf=zipf, layer=layer, dtype=dtype)

    max_data = num_classes * num_synonyms ** ((tuple_size**num_layers - 1) // (tuple_size - 1))
    if train_size < -1:
        raise ValueError("train_size must be >= -1.")

    if not replacement:
        if train_size == -1:
            samples = torch.arange(max_data, dtype=torch.long, device=dev)
            train_size_eff = max_data
            total = max_data
        else:
            train_size_eff = train_size
            test_size = min(test_size, max_data - train_size)
            total = train_size + test_size
            random.seed(int(seed_sample))
            samples_cpu = random.sample(range(max_data), total)
            samples = torch.tensor(samples_cpu, dtype=torch.long, device=dev)
        features, labels = sample_data_from_indices_torch(
            samples=samples,
            rules=rules,
            n=num_classes,
            m=num_synonyms,
            s=tuple_size,
            L=num_layers,
        )
        sample_ids = samples.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        torch.manual_seed(int(seed_sample))
        total = max_data + test_size if train_size == -1 else train_size + test_size
        train_size_eff = max_data if train_size == -1 else train_size
        labels = torch.randint(0, num_classes, size=(total,), dtype=torch.long, device=dev)
        if zipf is None:
            features, labels = sample_data_from_labels_torch(labels, rules)
        else:
            if layer is None:
                raise ValueError("zipf requires a selected layer.")
            prob = zipf_probabilities(num_synonyms, float(zipf), device=dev, dtype=dtype)
            features, labels = sample_data_from_labels_prob_torch(labels, rules, int(layer), prob)
        sample_ids = None

    train_sequences = features[:train_size_eff]
    test_sequences = features[train_size_eff : train_size_eff + test_size]
    train_labels = labels[:train_size_eff]
    test_labels = labels[train_size_eff : train_size_eff + test_size]

    return {
        "rules": rules,
        "rule_probs": rule_probs,
        "sample_ids": sample_ids,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "max_data": int(max_data),
    }


# -----------------------------------------------------------------------------
# Task creation: last-token and next-token prediction
# -----------------------------------------------------------------------------


def make_prediction_tasks(
    sequences: torch.Tensor,
    *,
    mode: str,
    q: int,
    positions: Optional[Sequence[int]] = None,
    max_tasks: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Convert full sequences [N,d] into masked observations [T,d].

    mode='last': one task per sequence, target position d-1.
    mode='next': one task per sequence-position, target positions 1,...,d-1
                 by default. Future tokens are masked/unobserved.

    Positions are zero-based token positions. For next-token prediction position 0
    is excluded because it has empty prefix.
    """
    if mode not in {"last", "next"}:
        raise ValueError("mode must be 'last' or 'next'.")
    seq = sequences.to(torch.long)
    dev = seq.device
    N, d = int(seq.shape[0]), int(seq.shape[1])
    mask_symbol = int(q)

    obs_list: List[torch.Tensor] = []
    target_pos: List[int] = []
    true_tokens: List[torch.Tensor] = []
    seq_indices: List[int] = []

    if mode == "last":
        obs = seq.clone()
        obs[:, -1] = mask_symbol
        return {
            "observations": obs,
            "target_pos": torch.full((N,), d - 1, dtype=torch.long, device=dev),
            "true_tokens": seq[:, -1].clone(),
            "seq_indices": torch.arange(N, dtype=torch.long, device=dev),
        }

    if positions is None:
        pos_list = list(range(1, d))
    else:
        pos_list = [int(p) for p in positions]
        if any(p <= 0 or p >= d for p in pos_list):
            raise ValueError(f"next-token positions must lie in [1,{d-1}].")

    for p in pos_list:
        obs = seq.clone()
        obs[:, p:] = mask_symbol  # target and future unobserved
        obs_list.append(obs)
        target_pos.extend([p] * N)
        true_tokens.append(seq[:, p].clone())
        seq_indices.extend(range(N))

    observations = torch.cat(obs_list, dim=0)
    target_pos_t = torch.tensor(target_pos, dtype=torch.long, device=dev)
    true_tokens_t = torch.cat(true_tokens, dim=0)
    seq_indices_t = torch.tensor(seq_indices, dtype=torch.long, device=dev)

    if max_tasks is not None and max_tasks < observations.shape[0]:
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        perm = torch.randperm(observations.shape[0], device=dev, generator=g)[: int(max_tasks)]
        observations = observations[perm]
        target_pos_t = target_pos_t[perm]
        true_tokens_t = true_tokens_t[perm]
        seq_indices_t = seq_indices_t[perm]

    return {
        "observations": observations,
        "target_pos": target_pos_t,
        "true_tokens": true_tokens_t,
        "seq_indices": seq_indices_t,
    }


# -----------------------------------------------------------------------------
# Batched torch BP with shared tau shrinkage
# -----------------------------------------------------------------------------


@dataclass
class BPStatsTorch:
    tau: torch.Tensor
    total_cost: torch.Tensor
    total_l2_norm: torch.Tensor
    num_penalized_messages: int
    converged_to_budget: Optional[torch.Tensor] = None


@dataclass
class BPBatchResult:
    marginals: List[torch.Tensor]
    context_messages: List[torch.Tensor]
    subtree_messages: List[torch.Tensor]
    stats: BPStatsTorch


def node_state_dim(depth: int, q: int, num_classes: int) -> int:
    return int(num_classes) if depth == 0 else int(q)


def encode_observations(
    observations: torch.Tensor,
    *,
    q: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """observations [B,d] with q meaning unobserved -> evidence [B,d,q]."""
    obs = observations.to(torch.long)
    B, d = obs.shape
    dev = obs.device
    leaves = torch.empty((B, d, q), dtype=dtype, device=dev)
    mask = obs == int(q)
    leaves[mask] = 1.0 / q
    if (~mask).any():
        leaves[~mask] = 0.0
        row, col = torch.where(~mask)
        leaves[row, col, obs[row, col]] = 1.0
    return leaves


def apply_tau_shrinkage_batch(
    msg: torch.Tensor,
    tau: torch.Tensor | float,
    *,
    eps: float = 1e-30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply c -> c/(1+tau) to a batch of messages [B,N,K].
    Returns shrunk message, cost per B, l2-sum per B.
    """
    msg = normalize(msg, dim=-1, eps=eps)
    c = centered_log_probs(msg, eps=eps)

    if not torch.is_tensor(tau):
        tau_t = torch.tensor(float(tau), dtype=msg.dtype, device=msg.device)
    else:
        tau_t = tau.to(dtype=msg.dtype, device=msg.device)

    if tau_t.ndim == 0:
        scale = 1.0 / (1.0 + tau_t)
        c_shrunk = c * scale
    else:
        # tau_t is [B]
        while tau_t.ndim < c.ndim:
            tau_t = tau_t.unsqueeze(-1)
        c_shrunk = c / (1.0 + tau_t)

    out = softmax_centered(c_shrunk)
    per_message_norm = torch.linalg.vector_norm(c_shrunk, ord=2, dim=-1)  # [B,N]
    cost_per_B = (per_message_norm**2).sum(dim=-1)
    l2_sum_per_B = per_message_norm.sum(dim=-1)
    return out, cost_per_B, l2_sum_per_B


def upward_candidate_message(
    child_block: torch.Tensor,
    rule_tensor: torch.Tensor,
    rule_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized upward BP candidate.

    child_block: [B,N,s,child_dim]
    rule_tensor: [parent_dim,m,s]
    rule_prob: [parent_dim,m]
    returns: [B,N,parent_dim]
    """
    B, N, s, child_dim = child_block.shape
    parent_dim, m, s2 = rule_tensor.shape
    assert s == s2
    prod_terms = rule_prob.to(child_block.dtype).unsqueeze(0).unsqueeze(0).expand(B, N, parent_dim, m)
    for t in range(s):
        idx = rule_tensor[:, :, t].to(child_block.device).unsqueeze(0).unsqueeze(0).expand(B, N, parent_dim, m)
        source = child_block[:, :, t, :].unsqueeze(2).expand(B, N, parent_dim, child_dim)
        gathered = torch.gather(source, dim=3, index=idx)
        prod_terms = prod_terms * gathered
    msg = prod_terms.sum(dim=-1)
    return normalize(msg, dim=-1)


def downward_candidate_for_child(
    parent_context: torch.Tensor,
    children_subtree: torch.Tensor,
    rule_tensor: torch.Tensor,
    rule_prob: torch.Tensor,
    child_pos: int,
) -> torch.Tensor:
    """
    Vectorized downward/context BP candidate for one child position.

    parent_context: [B,N,parent_dim]
    children_subtree: [B,N,s,child_dim]
    returns: [B,N,child_dim]
    """
    B, N, s, child_dim = children_subtree.shape
    parent_dim, m, _ = rule_tensor.shape
    weight = parent_context.unsqueeze(-1) * rule_prob.to(parent_context.dtype).unsqueeze(0).unsqueeze(0)
    # weight [B,N,parent_dim,m]
    for u in range(s):
        if u == child_pos:
            continue
        idx = rule_tensor[:, :, u].to(parent_context.device).unsqueeze(0).unsqueeze(0).expand(B, N, parent_dim, m)
        source = children_subtree[:, :, u, :].unsqueeze(2).expand(B, N, parent_dim, child_dim)
        sib = torch.gather(source, dim=3, index=idx)
        weight = weight * sib

    idx_child = rule_tensor[:, :, child_pos].to(parent_context.device).reshape(1, 1, parent_dim * m).expand(B, N, parent_dim * m)
    flat_weight = weight.reshape(B, N, parent_dim * m)
    out = torch.zeros((B, N, child_dim), dtype=parent_context.dtype, device=parent_context.device)
    out.scatter_add_(dim=2, index=idx_child, src=flat_weight)
    return normalize(out, dim=-1)


@torch.no_grad()
def bp_pass_torch(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    *,
    q: int,
    s: int,
    num_classes: int,
    tau: torch.Tensor | float = 0.0,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> BPBatchResult:
    """Run a full batched upward/downward BP pass with tau shrinkage."""
    dev = observations.device
    L = len(rules)
    B, d = observations.shape
    assert d == s**L, f"sequence length {d} must equal s**L={s**L}."

    subtree: List[torch.Tensor] = []
    context: List[torch.Tensor] = []
    leaves = encode_observations(observations, q=q, dtype=dtype)

    for depth in range(L + 1):
        K = node_state_dim(depth, q=q, num_classes=num_classes)
        n_nodes = s**depth
        subtree.append(torch.full((B, n_nodes, K), 1.0 / K, dtype=dtype, device=dev))
        context.append(torch.full((B, n_nodes, K), 1.0 / K, dtype=dtype, device=dev))
    subtree[L] = leaves

    total_cost = torch.zeros(B, dtype=dtype, device=dev)
    total_l2 = torch.zeros(B, dtype=dtype, device=dev)
    n_penalized = 0

    # Upward pass: leaves -> root.
    for depth in range(L - 1, -1, -1):
        child_nodes = subtree[depth + 1]
        n_nodes = s**depth
        child_block = child_nodes.reshape(B, n_nodes, s, -1)
        cand = upward_candidate_message(child_block, rules[depth], rule_probs[depth])
        msg, cost, l2 = apply_tau_shrinkage_batch(cand, tau)
        subtree[depth] = msg
        total_cost += cost
        total_l2 += l2
        n_penalized += n_nodes

    # Root context fixed prior, not penalized.
    if root_prior is None:
        root = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype, device=dev)
    else:
        root = normalize(root_prior.to(device=dev, dtype=dtype), dim=-1)
    context[0] = root.reshape(1, 1, -1).expand(B, 1, num_classes).clone()

    # Downward/context pass: root -> leaves.
    for depth in range(0, L):
        n_nodes = s**depth
        children_subtree = subtree[depth + 1].reshape(B, n_nodes, s, -1)
        out_children = torch.empty_like(children_subtree)
        for t in range(s):
            cand = downward_candidate_for_child(context[depth], children_subtree, rules[depth], rule_probs[depth], t)
            msg, cost, l2 = apply_tau_shrinkage_batch(cand, tau)
            out_children[:, :, t, :] = msg
            total_cost += cost
            total_l2 += l2
            n_penalized += n_nodes
        context[depth + 1] = out_children.reshape(B, n_nodes * s, -1)

    marginals: List[torch.Tensor] = []
    for depth in range(L + 1):
        prod = context[depth] * subtree[depth]
        marginals.append(normalize(prod, dim=-1))

    tau_tensor = tau if torch.is_tensor(tau) else torch.tensor(float(tau), dtype=dtype, device=dev)
    if tau_tensor.ndim == 0:
        tau_out = tau_tensor.expand(B).clone()
    else:
        tau_out = tau_tensor.to(dtype=dtype, device=dev).reshape(B)

    return BPBatchResult(
        marginals=marginals,
        context_messages=context,
        subtree_messages=subtree,
        stats=BPStatsTorch(
            tau=tau_out,
            total_cost=total_cost,
            total_l2_norm=total_l2,
            num_penalized_messages=n_penalized,
        ),
    )


@torch.no_grad()
def solve_tau_per_inference_batch(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    *,
    lambda_total: float,
    q: int,
    s: int,
    num_classes: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 40,
) -> Tuple[torch.Tensor, BPBatchResult]:
    """Vectorized per-inference tau bisection for one batch."""
    B = observations.shape[0]
    dev = observations.device
    lam = float(lambda_total)

    if not math.isfinite(lam):
        tau = torch.zeros(B, dtype=dtype, device=dev)
        return tau, bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=tau, root_prior=root_prior, dtype=dtype)

    if lam <= 0.0:
        tau = torch.full((B,), 1.0e12, dtype=dtype, device=dev)
        res = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=tau, root_prior=root_prior, dtype=dtype)
        res.stats.total_cost.zero_()
        res.stats.total_l2_norm.zero_()
        res.stats.converged_to_budget = torch.ones(B, dtype=torch.bool, device=dev)
        return tau, res

    res0 = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=torch.zeros(B, dtype=dtype, device=dev), root_prior=root_prior, dtype=dtype)
    already_ok = res0.stats.total_cost <= lam
    if bool(already_ok.all()):
        res0.stats.converged_to_budget = torch.ones(B, dtype=torch.bool, device=dev)
        return torch.zeros(B, dtype=dtype, device=dev), res0

    tau_lo = torch.zeros(B, dtype=dtype, device=dev)
    tau_hi = torch.ones(B, dtype=dtype, device=dev)
    # Keep tau=0 for already-unconstrained examples.
    tau_hi = torch.where(already_ok, torch.zeros_like(tau_hi), tau_hi)

    for _ in range(max_bisect_iter):
        res_hi = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=tau_hi, root_prior=root_prior, dtype=dtype)
        too_high = (res_hi.stats.total_cost > lam) & (~already_ok)
        if not bool(too_high.any()):
            break
        tau_hi = torch.where(too_high, tau_hi * 2.0, tau_hi)

    best_tau = tau_hi.clone()
    best_res = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=best_tau, root_prior=root_prior, dtype=dtype)
    best_gap = (best_res.stats.total_cost - lam).abs()

    for _ in range(max_bisect_iter):
        tau_mid = 0.5 * (tau_lo + tau_hi)
        tau_mid = torch.where(already_ok, torch.zeros_like(tau_mid), tau_mid)
        res_mid = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=tau_mid, root_prior=root_prior, dtype=dtype)
        gap = (res_mid.stats.total_cost - lam).abs()
        improve = gap < best_gap
        best_gap = torch.where(improve, gap, best_gap)
        best_tau = torch.where(improve, tau_mid, best_tau)
        # Cannot store partial BPBatchResult per-example; recompute best at end.

        too_high = (res_mid.stats.total_cost > lam) & (~already_ok)
        tau_lo = torch.where(too_high, tau_mid, tau_lo)
        tau_hi = torch.where(too_high, tau_hi, tau_mid)

        rel_gap = gap / max(lam, 1e-12)
        if bool(((rel_gap <= budget_tol_rel) | already_ok).all()) or bool(((tau_hi - tau_lo).abs() <= tau_tol).all()):
            break

    best_res = bp_pass_torch(rules, rule_probs, observations, q=q, s=s, num_classes=num_classes, tau=best_tau, root_prior=root_prior, dtype=dtype)
    best_res.stats.converged_to_budget = ((best_res.stats.total_cost - lam).abs() / max(lam, 1e-12) <= budget_tol_rel) | already_ok
    return best_tau, best_res


@torch.no_grad()
def solve_shared_tau_for_tasks(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    *,
    lambda_total: float,
    q: int,
    s: int,
    num_classes: int,
    batch_size: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 40,
) -> Tuple[float, float, bool]:
    """
    Find one shared tau so that the mean total message cost per prediction
    approximately equals lambda_total.
    """
    lam = float(lambda_total)
    if not math.isfinite(lam):
        return 0.0, float("nan"), True
    if lam <= 0.0:
        return 1.0e12, 0.0, True

    def mean_cost(tau_val: float) -> float:
        costs = []
        for start in range(0, observations.shape[0], batch_size):
            obs_b = observations[start : start + batch_size]
            res = bp_pass_torch(rules, rule_probs, obs_b, q=q, s=s, num_classes=num_classes, tau=float(tau_val), root_prior=root_prior, dtype=dtype)
            costs.append(res.stats.total_cost.detach())
        return float(torch.cat(costs).mean().item())

    c0 = mean_cost(0.0)
    if c0 <= lam:
        return 0.0, c0, True

    lo, hi = 0.0, 1.0
    chi = mean_cost(hi)
    expand_iter = 0
    while chi > lam and expand_iter < max_bisect_iter:
        hi *= 2.0
        chi = mean_cost(hi)
        expand_iter += 1

    best_tau, best_cost = hi, chi
    best_gap = abs(chi - lam)
    for _ in range(max_bisect_iter):
        mid = 0.5 * (lo + hi)
        cmid = mean_cost(mid)
        gap = abs(cmid - lam)
        if gap < best_gap:
            best_tau, best_cost, best_gap = mid, cmid, gap
        if gap / max(lam, 1e-12) <= budget_tol_rel or (hi - lo) <= tau_tol:
            return best_tau, best_cost, True
        if cmid > lam:
            lo = mid
        else:
            hi = mid
    return best_tau, best_cost, False


# -----------------------------------------------------------------------------
# Hierarchy masks for arbitrary next-token positions
# -----------------------------------------------------------------------------


def torch_rules_to_numpy(rules: Sequence[torch.Tensor]) -> List[np.ndarray]:
    return [r.detach().cpu().numpy().astype(np.int64, copy=False) for r in rules]


def possible_top_states_for_partial_block(
    block_values: np.ndarray,
    rules_slice: Sequence[np.ndarray],
    *,
    s: int,
    q: int,
) -> np.ndarray:
    """
    Return the possible top states of an aligned block under a partial leaf assignment.
    block_values entries: 0..q-1 observed, -1 unobserved.
    rules_slice is top-to-bottom for the block; we process it bottom-up.
    """
    possible: List[set[int]] = []
    for val in block_values.reshape(-1):
        if int(val) < 0:
            possible.append(set(range(q)))
        else:
            possible.append({int(val)})

    # Bottom-up: last rule tensor maps parent -> s child states at the current bottom level.
    for rule_tensor in reversed(rules_slice):
        parent_dim, m, s2 = rule_tensor.shape
        assert s2 == s
        if len(possible) % s != 0:
            return np.array([], dtype=np.int64)
        new_possible: List[set[int]] = []
        for j in range(len(possible) // s):
            child_sets = possible[j * s : (j + 1) * s]
            parents: set[int] = set()
            for parent in range(parent_dim):
                for r in range(m):
                    ok = True
                    for t in range(s):
                        if int(rule_tensor[parent, r, t]) not in child_sets[t]:
                            ok = False
                            break
                    if ok:
                        parents.add(parent)
                        break
            new_possible.append(parents)
        possible = new_possible
    if len(possible) != 1:
        return np.array([], dtype=np.int64)
    return np.array(sorted(possible[0]), dtype=np.int64)


def candidate_set_for_position_level(
    full_sequence: np.ndarray,
    target_pos: int,
    level: int,
    rules_np: Sequence[np.ndarray],
    *,
    s: int,
    q: int,
    L: int,
) -> np.ndarray:
    """
    Candidate set A_{i,level}: token values y such that the aligned level-block
    containing target_pos can be generated by the last `level` grammar layers,
    given the prefix observations inside that block and leaving future entries
    unobserved.
    """
    if not (1 <= level <= L):
        raise ValueError(f"level must lie in [1,{L}], got {level}.")
    block_size = s**level
    block_start = (int(target_pos) // block_size) * block_size
    block_end = block_start + block_size
    block = np.full(block_size, -1, dtype=np.int64)

    # Prefix observed within the block.
    for global_pos in range(block_start, min(target_pos, block_end)):
        block[global_pos - block_start] = int(full_sequence[global_pos])

    rules_slice = rules_np[L - level : L]
    out = np.zeros(q, dtype=bool)
    local_target = int(target_pos) - block_start
    for y in range(q):
        b = block.copy()
        b[local_target] = y
        out[y] = possible_top_states_for_partial_block(b, rules_slice, s=s, q=q).size > 0
    return out


def precompute_hierarchy_masks_for_tasks(
    sequences: torch.Tensor,
    tasks: Dict[str, torch.Tensor],
    rules: Sequence[torch.Tensor],
    *,
    s: int,
    q: int,
    L: int,
) -> Dict[str, np.ndarray]:
    """Precompute A/B masks for arbitrary target positions."""
    seq_np = sequences.detach().cpu().numpy().astype(np.int64, copy=False)
    seq_indices = tasks["seq_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
    target_pos = tasks["target_pos"].detach().cpu().numpy().astype(np.int64, copy=False)
    rules_np = torch_rules_to_numpy(rules)

    T = int(target_pos.shape[0])
    A_masks = np.zeros((T, L, q), dtype=bool)
    B_masks = np.zeros((T, L, q), dtype=bool)
    valid_masks = np.zeros((T, L), dtype=bool)
    all_vocab = np.ones(q, dtype=bool)

    for t in range(T):
        x = seq_np[seq_indices[t]]
        prev = all_vocab.copy()
        for ell in range(1, L + 1):
            A = candidate_set_for_position_level(
                x,
                int(target_pos[t]),
                ell,
                rules_np,
                s=s,
                q=q,
                L=L,
            )
            # Ensure nesting numerically/definitionally with previous level.
            A = A & prev
            B = prev & (~A)
            A_masks[t, ell - 1] = A
            B_masks[t, ell - 1] = B
            valid_masks[t, ell - 1] = bool(B.any() and A.any())
            prev = A
    return {"A_masks": A_masks, "B_masks": B_masks, "valid_masks": valid_masks}


# -----------------------------------------------------------------------------
# Evaluation loop and measures
# -----------------------------------------------------------------------------


def _init_level_accumulators(L: int) -> Dict[str, np.ndarray]:
    return {
        "A_mass_sum": np.zeros(L, dtype=np.float64),
        "B_mass_sum": np.zeros(L, dtype=np.float64),
        "margin_sum": np.zeros(L, dtype=np.float64),
        "margin_pos_count": np.zeros(L, dtype=np.float64),
        "hier_acc_count": np.zeros(L, dtype=np.float64),
        "penalty_sum": np.zeros(L, dtype=np.float64),
        "valid_count": np.zeros(L, dtype=np.float64),
        "penalty_all_sum": np.zeros(L, dtype=np.float64),
    }


def _update_hierarchy_accumulators(
    acc: Dict[str, np.ndarray],
    posterior: np.ndarray,
    pred: np.ndarray,
    true_tokens: np.ndarray,
    A_masks: np.ndarray,
    B_masks: np.ndarray,
    valid_masks: np.ndarray,
) -> Tuple[float, float]:
    """Update hierarchy accumulators. Returns residual_sum, reconstruction_abs_error_sum."""
    T, L, q = A_masks.shape
    residual_sum = 0.0
    recon_err_sum = 0.0
    for t in range(T):
        margins_for_recon = []
        for ell in range(L):
            A = A_masks[t, ell]
            B = B_masks[t, ell]
            valid = bool(valid_masks[t, ell])
            pA = float(posterior[t, A].sum()) if A.any() else 0.0
            pB = float(posterior[t, B].sum()) if B.any() else 0.0
            if valid:
                margin = float(np.log(max(pA, EPS)) - np.log(max(pB, EPS)))
                penalty = float(np.logaddexp(0.0, -margin))
                acc["A_mass_sum"][ell] += pA
                acc["B_mass_sum"][ell] += pB
                acc["margin_sum"][ell] += margin
                acc["margin_pos_count"][ell] += float(margin > 0.0)
                acc["hier_acc_count"][ell] += float(A[int(pred[t])])
                acc["penalty_sum"][ell] += penalty
                acc["valid_count"][ell] += 1.0
                margins_for_recon.append(margin)
                acc["penalty_all_sum"][ell] += penalty
            else:
                # No newly excluded set: this level contributes zero grouped penalty.
                margins_for_recon.append(float("inf"))
                acc["penalty_all_sum"][ell] += 0.0

        A_last = A_masks[t, -1]
        pAL = float(posterior[t, A_last].sum()) if A_last.any() else 0.0
        p_true = float(posterior[t, int(true_tokens[t])])
        residual = float(np.log(max(pAL, EPS)) - np.log(max(p_true, EPS)))
        residual_sum += residual
        penalties = [0.0 if np.isinf(m) else float(np.logaddexp(0.0, -m)) for m in margins_for_recon]
        reconstructed = residual + float(np.sum(penalties))
        actual_loss = float(-np.log(max(p_true, EPS)))
        recon_err_sum += abs(reconstructed - actual_loss)
    return residual_sum, recon_err_sum


@torch.no_grad()
def evaluate_tasks_for_lambda(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    tasks: Dict[str, torch.Tensor],
    hierarchy_masks: Dict[str, np.ndarray],
    *,
    lambda_total: float,
    prediction_mode: str,
    budget_scope: str,
    q: int,
    s: int,
    num_classes: int,
    batch_size: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 40,
) -> Dict[str, Any]:
    obs = tasks["observations"]
    target_pos = tasks["target_pos"]
    true_tokens = tasks["true_tokens"]
    T = int(obs.shape[0])
    L = len(rules)

    if budget_scope not in {"shared", "per_inference"}:
        raise ValueError("budget_scope must be 'shared' or 'per_inference'.")

    tau_shared: Optional[float] = None
    shared_budget_converged = True
    shared_measured_cost = float("nan")
    if budget_scope == "shared":
        tau_shared, shared_measured_cost, shared_budget_converged = solve_shared_tau_for_tasks(
            rules,
            rule_probs,
            obs,
            lambda_total=lambda_total,
            q=q,
            s=s,
            num_classes=num_classes,
            batch_size=batch_size,
            root_prior=root_prior,
            dtype=dtype,
            budget_tol_rel=budget_tol_rel,
            max_bisect_iter=max_bisect_iter,
        )

    losses: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    posterior_norms: List[np.ndarray] = []
    total_costs: List[np.ndarray] = []
    total_l2s: List[np.ndarray] = []
    taus: List[np.ndarray] = []
    converged: List[np.ndarray] = []

    acc = _init_level_accumulators(L)
    residual_sum = 0.0
    recon_err_sum = 0.0

    A_masks_all = hierarchy_masks["A_masks"]
    B_masks_all = hierarchy_masks["B_masks"]
    valid_masks_all = hierarchy_masks["valid_masks"]

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        obs_b = obs[start:end]
        target_b = target_pos[start:end]
        true_b = true_tokens[start:end]

        if budget_scope == "shared":
            res = bp_pass_torch(
                rules,
                rule_probs,
                obs_b,
                q=q,
                s=s,
                num_classes=num_classes,
                tau=float(tau_shared),
                root_prior=root_prior,
                dtype=dtype,
            )
            conv_b = torch.full((obs_b.shape[0],), bool(shared_budget_converged), dtype=torch.bool, device=obs_b.device)
        else:
            _, res = solve_tau_per_inference_batch(
                rules,
                rule_probs,
                obs_b,
                lambda_total=lambda_total,
                q=q,
                s=s,
                num_classes=num_classes,
                root_prior=root_prior,
                dtype=dtype,
                budget_tol_rel=budget_tol_rel,
                max_bisect_iter=max_bisect_iter,
            )
            conv_b = res.stats.converged_to_budget
            if conv_b is None:
                conv_b = torch.ones((obs_b.shape[0],), dtype=torch.bool, device=obs_b.device)

        leaf_marg = res.marginals[-1]  # [B,d,q]
        batch_indices = torch.arange(obs_b.shape[0], device=obs_b.device)
        posterior = leaf_marg[batch_indices, target_b, :]  # [B,q]
        pred = torch.argmax(posterior, dim=-1)
        p_true = posterior[batch_indices, true_b].clamp_min(EPS)
        loss = -torch.log(p_true)
        err = (pred != true_b).to(dtype)
        post_norm = centered_logit_l2_norm_torch(posterior)

        losses.append(loss.detach().cpu().numpy())
        errors.append(err.detach().cpu().numpy())
        posterior_norms.append(post_norm.detach().cpu().numpy())
        total_costs.append(res.stats.total_cost.detach().cpu().numpy())
        total_l2s.append(res.stats.total_l2_norm.detach().cpu().numpy())
        taus.append(res.stats.tau.detach().cpu().numpy())
        converged.append(conv_b.detach().cpu().numpy().astype(np.float64))

        posterior_np = posterior.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy().astype(np.int64, copy=False)
        true_np = true_b.detach().cpu().numpy().astype(np.int64, copy=False)
        rs, re = _update_hierarchy_accumulators(
            acc,
            posterior_np,
            pred_np,
            true_np,
            A_masks_all[start:end],
            B_masks_all[start:end],
            valid_masks_all[start:end],
        )
        residual_sum += rs
        recon_err_sum += re

    losses_np = np.concatenate(losses)
    errors_np = np.concatenate(errors)
    posterior_norms_np = np.concatenate(posterior_norms)
    total_costs_np = np.concatenate(total_costs)
    total_l2s_np = np.concatenate(total_l2s)
    taus_np = np.concatenate(taus)
    converged_np = np.concatenate(converged)

    valid_count = acc["valid_count"]
    denom_valid = np.maximum(valid_count, 1.0)

    return {
        "lambda": float(lambda_total),
        "prediction_mode": prediction_mode,
        "budget_scope": budget_scope,
        "loss_mean": float(losses_np.mean()),
        "loss_std": float(losses_np.std()),
        "error_mean": float(errors_np.mean()),
        "error_std": float(errors_np.std()),
        "posterior_norm_mean": float(posterior_norms_np.mean()),
        "posterior_norm_std": float(posterior_norms_np.std()),
        "message_total_cost_mean": float(total_costs_np.mean()),
        "message_total_cost_std": float(total_costs_np.std()),
        "message_total_l2_norm_mean": float(total_l2s_np.mean()),
        "message_total_l2_norm_std": float(total_l2s_np.std()),
        "tau_mean": float(taus_np.mean()),
        "tau_std": float(taus_np.std()),
        "tau_shared": None if tau_shared is None else float(tau_shared),
        "shared_measured_cost": None if tau_shared is None else float(shared_measured_cost),
        "budget_hit_fraction": float(converged_np.mean()),
        "num_tasks": int(T),
        "num_penalized_messages": int(res.stats.num_penalized_messages),
        "A_mass_mean": acc["A_mass_sum"] / denom_valid,
        "B_mass_mean": acc["B_mass_sum"] / denom_valid,
        "margin_mean": acc["margin_sum"] / denom_valid,
        "margin_pos_frac": acc["margin_pos_count"] / denom_valid,
        "hier_acc": acc["hier_acc_count"] / denom_valid,
        "level_penalty_mean": acc["penalty_sum"] / denom_valid,
        "level_penalty_all_mean": acc["penalty_all_sum"] / float(T),
        "valid_level_frac": valid_count / float(T),
        "residual_mean": float(residual_sum / float(T)),
        "reconstructed_loss_abs_error_mean": float(recon_err_sum / float(T)),
    }


def _parse_lambda_values_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.lambda_values is not None:
        return np.array([float(x) for x in args.lambda_values.split(",") if x.strip()], dtype=np.float64)
    vals = np.logspace(args.lambda_log10_min, args.lambda_log10_max, args.lambda_num)
    if args.include_zero:
        vals = np.concatenate(([0.0], vals))
    if args.include_inf:
        vals = np.concatenate((vals, [np.inf]))
    return vals.astype(np.float64)


def _stack_sweep_array(sweep: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = [r[key] for r in sweep]
    if isinstance(vals[0], np.ndarray):
        return np.stack(vals, axis=0)
    return np.asarray(vals)


def simulate_constrained_bp_sweep(
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
    prediction_mode: str = "last",
    budget_scope: str = "shared",
    positions: Optional[Sequence[int]] = None,
    max_test_sequences: Optional[int] = None,
    max_tasks: Optional[int] = None,
    task_seed: int = 0,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    budget_tol_rel: float = 1e-6,
    max_bisect_iter: int = 100,
) -> Dict[str, Any]:
    """
    Main notebook-friendly function.

    Returns a dictionary with arrays indexed by lambda_values.
    """
    dev = _as_device(device)
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-2, 2, 25), [np.inf]))
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
        device=dev,
        dtype=dtype,
    )

    test_sequences = data["test_sequences"]
    if max_test_sequences is not None:
        test_sequences = test_sequences[: int(max_test_sequences)]
    if test_sequences.numel() == 0:
        raise ValueError("No test sequences are available.")

    tasks = make_prediction_tasks(
        test_sequences,
        mode=prediction_mode,
        q=num_features,
        positions=positions,
        max_tasks=max_tasks,
        seed=task_seed,
    )

    hierarchy_masks = precompute_hierarchy_masks_for_tasks(
        test_sequences,
        tasks,
        data["rules"],
        s=tuple_size,
        q=num_features,
        L=num_layers,
    )

    sweep: List[Dict[str, Any]] = []
    iterator = tqdm(lambda_values, desc=f"{prediction_mode} BP sweep ({budget_scope})")
    for lam in iterator:
        out = evaluate_tasks_for_lambda(
            data["rules"],
            data["rule_probs"],
            tasks,
            hierarchy_masks,
            lambda_total=float(lam),
            prediction_mode=prediction_mode,
            budget_scope=budget_scope,
            q=num_features,
            s=tuple_size,
            num_classes=num_classes,
            batch_size=batch_size,
            dtype=dtype,
            budget_tol_rel=budget_tol_rel,
            max_bisect_iter=max_bisect_iter,
        )
        sweep.append(out)

    result = {
        "params": {
            "num_features": int(num_features),
            "num_classes": int(num_classes),
            "num_synonyms": int(num_synonyms),
            "tuple_size": int(tuple_size),
            "num_layers": int(num_layers),
            "train_size": int(train_size),
            "test_size": int(test_size),
            "seed_rules": int(seed_rules),
            "seed_sample": int(seed_sample),
            "prediction_mode": str(prediction_mode),
            "budget_scope": str(budget_scope),
            "positions": None if positions is None else [int(p) for p in positions],
            "max_test_sequences": None if max_test_sequences is None else int(max_test_sequences),
            "max_tasks": None if max_tasks is None else int(max_tasks),
            "task_seed": int(task_seed),
            "zipf": None if zipf is None else float(zipf),
            "layer": None if layer is None else int(layer),
            "replacement": None if replacement is None else bool(replacement),
            "last_layer_powerlaw_a": None if last_layer_powerlaw_a is None else float(last_layer_powerlaw_a),
            "batch_size": int(batch_size),
            "device": str(dev),
            "dtype": str(dtype).replace("torch.", ""),
            "budget_tol_rel": float(budget_tol_rel),
            "max_bisect_iter": int(max_bisect_iter),
        },
        "lambda_values": lambda_values,
        "loss_mean": _stack_sweep_array(sweep, "loss_mean"),
        "loss_std": _stack_sweep_array(sweep, "loss_std"),
        "error_mean": _stack_sweep_array(sweep, "error_mean"),
        "error_std": _stack_sweep_array(sweep, "error_std"),
        "posterior_norm_mean": _stack_sweep_array(sweep, "posterior_norm_mean"),
        "posterior_norm_std": _stack_sweep_array(sweep, "posterior_norm_std"),
        "message_total_cost_mean": _stack_sweep_array(sweep, "message_total_cost_mean"),
        "message_total_cost_std": _stack_sweep_array(sweep, "message_total_cost_std"),
        "message_total_l2_norm_mean": _stack_sweep_array(sweep, "message_total_l2_norm_mean"),
        "message_total_l2_norm_std": _stack_sweep_array(sweep, "message_total_l2_norm_std"),
        "tau_mean": _stack_sweep_array(sweep, "tau_mean"),
        "tau_std": _stack_sweep_array(sweep, "tau_std"),
        "tau_shared": _stack_sweep_array(sweep, "tau_shared"),
        "shared_measured_cost": _stack_sweep_array(sweep, "shared_measured_cost"),
        "budget_hit_fraction": _stack_sweep_array(sweep, "budget_hit_fraction"),
        "A_mass_mean": _stack_sweep_array(sweep, "A_mass_mean"),
        "B_mass_mean": _stack_sweep_array(sweep, "B_mass_mean"),
        "margin_mean": _stack_sweep_array(sweep, "margin_mean"),
        "margin_pos_frac": _stack_sweep_array(sweep, "margin_pos_frac"),
        "hier_acc": _stack_sweep_array(sweep, "hier_acc"),
        "level_penalty_mean": _stack_sweep_array(sweep, "level_penalty_mean"),
        "level_penalty_all_mean": _stack_sweep_array(sweep, "level_penalty_all_mean"),
        "valid_level_frac": _stack_sweep_array(sweep, "valid_level_frac"),
        "residual_mean": _stack_sweep_array(sweep, "residual_mean"),
        "reconstructed_loss_abs_error_mean": _stack_sweep_array(sweep, "reconstructed_loss_abs_error_mean"),
        "raw_per_lambda": sweep,
        "test_sequences": test_sequences.detach().cpu().numpy().astype(np.int64, copy=False),
        "task_target_pos": tasks["target_pos"].detach().cpu().numpy().astype(np.int64, copy=False),
        "task_true_tokens": tasks["true_tokens"].detach().cpu().numpy().astype(np.int64, copy=False),
        "task_seq_indices": tasks["seq_indices"].detach().cpu().numpy().astype(np.int64, copy=False),
        "A_masks": hierarchy_masks["A_masks"],
        "B_masks": hierarchy_masks["B_masks"],
        "valid_masks": hierarchy_masks["valid_masks"],
        "note": (
            "Torch global-budget BP sweep for last-token or next-token prediction. "
            "lambda_values are target total centered-log-message costs per prediction. "
            "budget_scope='shared' fits one tau for the whole evaluation set so the mean cost matches lambda; "
            "budget_scope='per_inference' fits one tau per prediction problem, closer to the older code. "
            "Next-token tasks clamp the prefix, mask target and future tokens, and read the target-leaf marginal. "
            "Hierarchy observables use aligned level-block compatible sets A_{i,l}, B_{i,l}; levels with empty B are masked in margin averages."
        ),
    }
    return result


def save_results_npz(results: Dict[str, Any], out_prefix: str | Path) -> Path:
    out_prefix = Path(out_prefix)
    path = Path(str(out_prefix) + ".npz")
    np.savez_compressed(
        path,
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
        tau_shared=results["tau_shared"],
        shared_measured_cost=results["shared_measured_cost"],
        budget_hit_fraction=results["budget_hit_fraction"],
        A_mass_mean=results["A_mass_mean"],
        B_mass_mean=results["B_mass_mean"],
        margin_mean=results["margin_mean"],
        margin_pos_frac=results["margin_pos_frac"],
        hier_acc=results["hier_acc"],
        level_penalty_mean=results["level_penalty_mean"],
        level_penalty_all_mean=results["level_penalty_all_mean"],
        valid_level_frac=results["valid_level_frac"],
        residual_mean=results["residual_mean"],
        reconstructed_loss_abs_error_mean=results["reconstructed_loss_abs_error_mean"],
        test_sequences=results["test_sequences"],
        task_target_pos=results["task_target_pos"],
        task_true_tokens=results["task_true_tokens"],
        task_seq_indices=results["task_seq_indices"],
        A_masks=results["A_masks"],
        B_masks=results["B_masks"],
        valid_masks=results["valid_masks"],
        params_json=json.dumps(results["params"], sort_keys=True),
        note=results["note"],
    )
    return path


def _dtype_from_string(name: str) -> torch.dtype:
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError("dtype must be float64 or float32.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch constrained BP sweep for RHM last/next-token prediction.")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=2048)
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--prediction_mode", type=str, default="last", choices=["last", "next"])
    parser.add_argument("--budget_scope", type=str, default="shared", choices=["shared", "per_inference"])
    parser.add_argument("--positions", type=str, default=None, help="Comma-separated zero-based target positions for next mode, e.g. '1,2,3'.")
    parser.add_argument("--max_test_sequences", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--task_seed", type=int, default=0)
    parser.add_argument("--zipf", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None)
    parser.add_argument("--lambda_values", type=str, default=None, help="Comma-separated lambda values. Use 'inf' for exact BP.")
    parser.add_argument("--lambda_log10_min", type=float, default=-2.0)
    parser.add_argument("--lambda_log10_max", type=float, default=2.0)
    parser.add_argument("--lambda_num", type=int, default=25)
    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--include_inf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--budget_tol_rel", type=float, default=1e-4)
    parser.add_argument("--max_bisect_iter", type=int, default=40)
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/constrained_bp_last_next_torch")
    args = parser.parse_args()

    lambda_values = _parse_lambda_values_from_args(args)
    positions = None
    if args.positions is not None:
        positions = [int(x) for x in args.positions.split(",") if x.strip()]

    results = simulate_constrained_bp_sweep(
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
        prediction_mode=args.prediction_mode,
        budget_scope=args.budget_scope,
        positions=positions,
        max_test_sequences=args.max_test_sequences,
        max_tasks=args.max_tasks,
        task_seed=args.task_seed,
        zipf=args.zipf,
        layer=args.layer,
        replacement=True if args.replacement else None,
        last_layer_powerlaw_a=args.last_layer_powerlaw_a,
        batch_size=args.batch_size,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
        budget_tol_rel=args.budget_tol_rel,
        max_bisect_iter=args.max_bisect_iter,
    )

    path = save_results_npz(results, args.out_prefix)
    print("Saved", path)
    print(results["note"])
    print("params:", json.dumps(results["params"], indent=2, sort_keys=True))
    print("lambda_values:", results["lambda_values"])
    print("message_total_cost_mean:", results["message_total_cost_mean"])
    print("tau_mean:", results["tau_mean"])
    print("loss_mean:", results["loss_mean"])
    print("error_mean:", results["error_mean"])
    print("margin_mean:\n", results["margin_mean"])
    print("margin_pos_frac:\n", results["margin_pos_frac"])
    print("valid_level_frac:\n", results["valid_level_frac"])
    print("level_penalty_mean:\n", results["level_penalty_mean"])
    print("reconstructed_loss_abs_error_mean:", results["reconstructed_loss_abs_error_mean"])


if __name__ == "__main__":
    main()
