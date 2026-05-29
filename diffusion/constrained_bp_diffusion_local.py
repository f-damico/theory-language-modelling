#!/usr/bin/env python3
"""
Local-norm constrained oracle BP for masked/discrete diffusion on the RHM.

Goal
----
This file is the diffusion analogue of the last-token / next-token constrained
BP scripts.  It evaluates, for a grid of local message radii lambda,

    L_BP(lambda) = E[-log p_lambda(x0_i | x_t, t)]

where p_lambda is obtained by BP on the known RHM tree, but every internal BP
message is projected to a centered-logit L2 ball of radius lambda.

The default is the "full-noise" masked-diffusion slice: all leaves are masked
and all positions are scored.  More generally, you can pass one or more mask
probabilities and the script averages over the resulting corrupted test tasks,
so the output is still one scalar loss for each lambda.

Outputs include the exact level-wise margin decomposition of the denoising
cross-entropy:

    -log p_lambda(x0_i | x_t) = R + sum_l log(1 + exp(-M_l))

with, for each level, the mean margin, the mean peeled loss, and the fraction
of negative margins Pr[M_l < 0].

Notebook usage
--------------
    import constrained_bp_diffusion_local as bpd
    res = bpd.simulate_local_bp_diffusion_sweep(
        num_features=16, num_classes=16, num_synonyms=4,
        tuple_size=2, num_layers=3,
        test_size=512,
        lambda_values=[0, 0.1, 0.3, 1, 3, 10, float('inf')],
        mask_prob_values=[1.0],   # full noise
        device='cuda',
    )
    bpd.save_results_npz(res, 'bp_diffusion_local_L3')

Terminal usage
--------------
    python constrained_bp_diffusion_local.py \
        --num_features 16 --num_classes 16 --num_synonyms 4 \
        --tuple_size 2 --num_layers 3 --test_size 1024 \
        --mask_prob_values 1.0 --include_zero --include_inf \
        --lambda_log10_min -2 --lambda_log10_max 1.5 --lambda_num 20 \
        --device cuda --out_prefix results/bp_diffusion_local_L3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic tensor utilities
# -----------------------------------------------------------------------------


def _as_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return dev


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-30) -> torch.Tensor:
    total = x.sum(dim=dim, keepdim=True)
    bad = (~torch.isfinite(total)) | (total <= eps)
    if bad.any():
        x = x.clone()
        x[bad.expand_as(x)] = 1.0
        total = x.sum(dim=dim, keepdim=True)
    return x / total.clamp_min(eps)


def centered_log_probs(prob: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    prob = normalize(prob, dim=-1, eps=eps)
    logp = torch.log(prob.clamp_min(eps))
    return logp - logp.mean(dim=-1, keepdim=True)


def softmax_centered(c: torch.Tensor) -> torch.Tensor:
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
# RHM construction and data sampling, same convention as the previous scripts
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
    features = labels.to(torch.long).reshape(-1, 1)
    for level_rules in rules:
        chosen_rule = torch.randint(0, int(level_rules.shape[1]), size=features.shape, device=features.device)
        features = level_rules[features, chosen_rule].flatten(start_dim=1)
    return features, labels


def sample_data_from_labels_prob_torch(
    labels: torch.Tensor,
    rules: Sequence[torch.Tensor],
    layer: int,
    prob: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        else:
            train_size_eff = train_size
            test_size = min(test_size, max_data - train_size)
            total = train_size + test_size
            random.seed(int(seed_sample))
            samples = torch.tensor(random.sample(range(max_data), total), dtype=torch.long, device=dev)
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

    return {
        "rules": rules,
        "rule_probs": rule_probs,
        "sample_ids": sample_ids,
        "train_sequences": features[:train_size_eff],
        "test_sequences": features[train_size_eff : train_size_eff + test_size],
        "train_labels": labels[:train_size_eff],
        "test_labels": labels[train_size_eff : train_size_eff + test_size],
        "max_data": int(max_data),
    }


# -----------------------------------------------------------------------------
# Local-constrained batched BP
# -----------------------------------------------------------------------------


@dataclass
class BPStatsLocal:
    total_cost: torch.Tensor
    total_l2_norm: torch.Tensor
    max_message_norm: torch.Tensor
    num_penalized_messages: int
    clipped_fraction: torch.Tensor


@dataclass
class BPBatchResultLocal:
    marginals: List[torch.Tensor]
    context_messages: List[torch.Tensor]
    subtree_messages: List[torch.Tensor]
    stats: BPStatsLocal


def node_state_dim(depth: int, q: int, num_classes: int) -> int:
    return int(num_classes) if depth == 0 else int(q)


def encode_observations(observations: torch.Tensor, *, q: int, dtype: torch.dtype) -> torch.Tensor:
    """observations [B,d], q means unobserved/masked -> evidence [B,d,q]."""
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


def project_local_centered_logit_ball_batch(
    msg: torch.Tensor,
    lambda_radius: float,
    *,
    eps: float = 1e-30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project each message independently to ||c||_2 <= lambda_radius, where
    c = log msg - mean(log msg).  msg has shape [B,N,K].

    Returns projected message, cost per observation, l2-sum per observation,
    max message norm per observation, and clipped-count per observation.
    """
    msg = normalize(msg, dim=-1, eps=eps)
    B, N, K = msg.shape
    dev = msg.device
    dtype = msg.dtype

    if math.isinf(float(lambda_radius)):
        c = centered_log_probs(msg, eps=eps)
        norms = torch.linalg.vector_norm(c, ord=2, dim=-1)
        return msg, (norms**2).sum(dim=-1), norms.sum(dim=-1), norms.max(dim=-1).values, torch.zeros(B, dtype=dtype, device=dev)

    lam = max(float(lambda_radius), 0.0)
    if lam <= 0.0:
        out = torch.full_like(msg, 1.0 / K)
        zeros = torch.zeros(B, dtype=dtype, device=dev)
        # Messages are uniform after projection; all non-uniform candidates are effectively clipped.
        c0 = centered_log_probs(msg, eps=eps)
        n0 = torch.linalg.vector_norm(c0, ord=2, dim=-1)
        clipped = (n0 > 1e-14).to(dtype).sum(dim=-1)
        return out, zeros, zeros, zeros, clipped

    c = centered_log_probs(msg, eps=eps)
    norms = torch.linalg.vector_norm(c, ord=2, dim=-1)
    scale = torch.clamp(torch.tensor(lam, dtype=dtype, device=dev) / norms.clamp_min(eps), max=1.0)
    c_proj = c * scale.unsqueeze(-1)
    out = softmax_centered(c_proj)
    norms_proj = norms * scale
    clipped = (scale < 1.0 - 1e-12).to(dtype).sum(dim=-1)
    return out, (norms_proj**2).sum(dim=-1), norms_proj.sum(dim=-1), norms_proj.max(dim=-1).values, clipped


def upward_candidate_message(
    child_block: torch.Tensor,
    rule_tensor: torch.Tensor,
    rule_prob: torch.Tensor,
) -> torch.Tensor:
    B, N, s, child_dim = child_block.shape
    parent_dim, m, s2 = rule_tensor.shape
    assert s == s2
    prod_terms = rule_prob.to(child_block.dtype).unsqueeze(0).unsqueeze(0).expand(B, N, parent_dim, m)
    for t in range(s):
        idx = rule_tensor[:, :, t].to(child_block.device).unsqueeze(0).unsqueeze(0).expand(B, N, parent_dim, m)
        source = child_block[:, :, t, :].unsqueeze(2).expand(B, N, parent_dim, child_dim)
        gathered = torch.gather(source, dim=3, index=idx)
        prod_terms = prod_terms * gathered
    return normalize(prod_terms.sum(dim=-1), dim=-1)


def downward_candidate_for_child(
    parent_context: torch.Tensor,
    children_subtree: torch.Tensor,
    rule_tensor: torch.Tensor,
    rule_prob: torch.Tensor,
    child_pos: int,
) -> torch.Tensor:
    B, N, s, child_dim = children_subtree.shape
    parent_dim, m, _ = rule_tensor.shape
    weight = parent_context.unsqueeze(-1) * rule_prob.to(parent_context.dtype).unsqueeze(0).unsqueeze(0)
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
def bp_pass_local_torch(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    *,
    lambda_radius: float,
    q: int,
    s: int,
    num_classes: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> BPBatchResultLocal:
    """Run one batched BP pass with a local centered-logit radius on each message."""
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
    max_norm = torch.zeros(B, dtype=dtype, device=dev)
    clipped_count = torch.zeros(B, dtype=dtype, device=dev)
    n_penalized = 0

    # Upward pass.
    for depth in range(L - 1, -1, -1):
        n_nodes = s**depth
        child_block = subtree[depth + 1].reshape(B, n_nodes, s, -1)
        cand = upward_candidate_message(child_block, rules[depth], rule_probs[depth])
        msg, cost, l2, mx, clipped = project_local_centered_logit_ball_batch(cand, lambda_radius)
        subtree[depth] = msg
        total_cost += cost
        total_l2 += l2
        max_norm = torch.maximum(max_norm, mx)
        clipped_count += clipped
        n_penalized += n_nodes

    # Root prior is not penalized.
    if root_prior is None:
        root = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype, device=dev)
    else:
        root = normalize(root_prior.to(device=dev, dtype=dtype), dim=-1)
    context[0] = root.reshape(1, 1, -1).expand(B, 1, num_classes).clone()

    # Downward pass.
    for depth in range(0, L):
        n_nodes = s**depth
        children_subtree = subtree[depth + 1].reshape(B, n_nodes, s, -1)
        out_children = torch.empty_like(children_subtree)
        for t in range(s):
            cand = downward_candidate_for_child(context[depth], children_subtree, rules[depth], rule_probs[depth], t)
            msg, cost, l2, mx, clipped = project_local_centered_logit_ball_batch(cand, lambda_radius)
            out_children[:, :, t, :] = msg
            total_cost += cost
            total_l2 += l2
            max_norm = torch.maximum(max_norm, mx)
            clipped_count += clipped
            n_penalized += n_nodes
        context[depth + 1] = out_children.reshape(B, n_nodes * s, -1)

    marginals: List[torch.Tensor] = []
    for depth in range(L + 1):
        marginals.append(normalize(context[depth] * subtree[depth], dim=-1))

    return BPBatchResultLocal(
        marginals=marginals,
        context_messages=context,
        subtree_messages=subtree,
        stats=BPStatsLocal(
            total_cost=total_cost,
            total_l2_norm=total_l2,
            max_message_norm=max_norm,
            num_penalized_messages=n_penalized,
            clipped_fraction=clipped_count / max(float(n_penalized), 1.0),
        ),
    )


# -----------------------------------------------------------------------------
# Masked-diffusion task construction
# -----------------------------------------------------------------------------


def make_masked_diffusion_observations(
    sequences: torch.Tensor,
    *,
    q: int,
    mask_prob_values: Sequence[float] = (1.0,),
    num_corruptions_per_sequence: int = 1,
    target_mode: str = "masked_only",
    seed: int = 0,
    force_nonempty_targets: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build corrupted observations for masked diffusion.

    target_mode:
      - 'masked_only': the loss is averaged only over masked/noised positions.
      - 'all': the loss is averaged over every position.

    The default with mask_prob_values=[1.0] is full noise: all positions are
    masked and all positions are scored.
    """
    if target_mode not in {"masked_only", "all"}:
        raise ValueError("target_mode must be 'masked_only' or 'all'.")

    seq = sequences.to(torch.long)
    dev = seq.device
    N, d = int(seq.shape[0]), int(seq.shape[1])
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    obs_list: List[torch.Tensor] = []
    clean_list: List[torch.Tensor] = []
    target_mask_list: List[torch.Tensor] = []
    p_list: List[torch.Tensor] = []

    for p_raw in mask_prob_values:
        p = float(p_raw)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"mask probabilities must lie in [0,1], got {p}.")
        for _ in range(int(num_corruptions_per_sequence)):
            if p >= 1.0:
                mask = torch.ones((N, d), dtype=torch.bool, device=dev)
            elif p <= 0.0:
                mask = torch.zeros((N, d), dtype=torch.bool, device=dev)
            else:
                mask = torch.rand((N, d), generator=gen, device=dev) < p
                if target_mode == "masked_only" and force_nonempty_targets:
                    empty = ~mask.any(dim=1)
                    if bool(empty.any()):
                        pos = torch.randint(0, d, size=(int(empty.sum().item()),), generator=gen, device=dev)
                        rows = torch.where(empty)[0]
                        mask[rows, pos] = True

            obs = seq.clone()
            obs[mask] = int(q)
            target_mask = mask if target_mode == "masked_only" else torch.ones_like(mask)
            # If p=0 and target_mode=masked_only, there are no targets unless forced above.
            if target_mask.any():
                obs_list.append(obs)
                clean_list.append(seq.clone())
                target_mask_list.append(target_mask)
                p_list.append(torch.full((N,), p, dtype=torch.float64, device=dev))

    if not obs_list:
        raise ValueError("No diffusion observations/targets were generated. Check mask_prob_values and target_mode.")

    return {
        "observations": torch.cat(obs_list, dim=0),
        "clean_sequences": torch.cat(clean_list, dim=0),
        "target_mask": torch.cat(target_mask_list, dim=0),
        "mask_prob_per_observation": torch.cat(p_list, dim=0),
    }


# -----------------------------------------------------------------------------
# Hierarchy masks A_l/B_l for arbitrary masked-diffusion observations
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
    Return possible top states of an aligned block under partial leaf evidence.
    block_values entries are 0..q-1 for observed leaves and -1 for unobserved.
    rules_slice is top-to-bottom; internally it is processed bottom-up.
    """
    possible: List[set[int]] = []
    for val in block_values.reshape(-1):
        if int(val) < 0:
            possible.append(set(range(q)))
        else:
            possible.append({int(val)})

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
                    for u in range(s):
                        if int(rule_tensor[parent, r, u]) not in child_sets[u]:
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


def candidate_set_for_diffusion_position_level(
    observation: np.ndarray,
    target_pos: int,
    level: int,
    rules_np: Sequence[np.ndarray],
    *,
    s: int,
    q: int,
    L: int,
) -> np.ndarray:
    """
    A_{i,level}: token values y such that the aligned level-block containing
    target_pos can be generated by the last `level` grammar layers, given the
    currently observed clean tokens in x_t and treating masked tokens as unknown.
    The target position is tested as y even if it is masked in the observation.
    """
    if not (1 <= level <= L):
        raise ValueError(f"level must lie in [1,{L}], got {level}.")
    block_size = s**level
    block_start = (int(target_pos) // block_size) * block_size
    block_end = block_start + block_size
    obs_block = np.asarray(observation[block_start:block_end], dtype=np.int64)
    block = np.where(obs_block == int(q), -1, obs_block).astype(np.int64, copy=True)
    local_target = int(target_pos) - block_start

    rules_slice = rules_np[L - level : L]
    out = np.zeros(q, dtype=bool)
    for y in range(q):
        b = block.copy()
        b[local_target] = int(y)
        out[y] = possible_top_states_for_partial_block(b, rules_slice, s=s, q=q).size > 0
    return out


def precompute_hierarchy_masks_for_diffusion_targets(
    observations: torch.Tensor,
    clean_sequences: torch.Tensor,
    target_mask: torch.Tensor,
    rules: Sequence[torch.Tensor],
    *,
    s: int,
    q: int,
    L: int,
    margin_context: str = "clean",
) -> Dict[str, np.ndarray]:
    if margin_context not in {"clean", "observed"}:
        raise ValueError("margin_context must be 'clean' or 'observed'.")
    obs_np = observations.detach().cpu().numpy().astype(np.int64, copy=False)
    clean_np = clean_sequences.detach().cpu().numpy().astype(np.int64, copy=False)
    target_np = target_mask.detach().cpu().numpy().astype(bool, copy=False)
    rules_np = torch_rules_to_numpy(rules)

    flat_obs_index: List[int] = []
    flat_target_pos: List[int] = []
    flat_true_tokens: List[int] = []
    for r in range(obs_np.shape[0]):
        positions = np.where(target_np[r])[0]
        for pos in positions:
            flat_obs_index.append(r)
            flat_target_pos.append(int(pos))
            flat_true_tokens.append(int(clean_np[r, pos]))

    T = len(flat_obs_index)
    A_masks = np.zeros((T, L, q), dtype=bool)
    B_masks = np.zeros((T, L, q), dtype=bool)
    valid_masks = np.zeros((T, L), dtype=bool)
    all_vocab = np.ones(q, dtype=bool)

    for t in tqdm(range(T), desc="precompute A/B masks", leave=False):
        row = flat_obs_index[t]
        obs = clean_np[row] if margin_context == "clean" else obs_np[row]
        target_pos = flat_target_pos[t]
        prev = all_vocab.copy()
        for ell in range(1, L + 1):
            A = candidate_set_for_diffusion_position_level(
                obs,
                target_pos,
                ell,
                rules_np,
                s=s,
                q=q,
                L=L,
            )
            A = A & prev  # enforce nesting numerically
            B = prev & (~A)
            A_masks[t, ell - 1] = A
            B_masks[t, ell - 1] = B
            valid_masks[t, ell - 1] = bool(A.any() and B.any())
            prev = A

    return {
        "A_masks": A_masks,
        "B_masks": B_masks,
        "valid_masks": valid_masks,
        "flat_obs_index": np.asarray(flat_obs_index, dtype=np.int64),
        "flat_target_pos": np.asarray(flat_target_pos, dtype=np.int64),
        "flat_true_tokens": np.asarray(flat_true_tokens, dtype=np.int64),
    }


# -----------------------------------------------------------------------------
# Grammar-validity utilities for generated RHM strings
# -----------------------------------------------------------------------------


def _rules_sequence_to_numpy_list(rules: Sequence[torch.Tensor | np.ndarray] | Dict[int, Any]) -> List[np.ndarray]:
    """
    Convert rules to a top-to-bottom list of numpy arrays.

    Convention inherited from the RHM code:
      rules[0]      : root/top production table, shape [num_classes, m, s]
      rules[1..L-1] : lower hidden production tables, shape [q, m, s]

    The bottom leaf-level production table is rules[L-1].
    """
    if isinstance(rules, dict):
        keys = sorted(int(k) for k in rules.keys())
        seq = [rules[k] for k in keys]
    else:
        seq = list(rules)
    out: List[np.ndarray] = []
    for r in seq:
        if torch.is_tensor(r):
            arr = r.detach().cpu().numpy()
        else:
            arr = np.asarray(r)
        out.append(np.asarray(arr, dtype=np.int64))
    return out


def _inverse_maps_from_rules(rules: Sequence[torch.Tensor | np.ndarray] | Dict[int, Any]) -> List[Dict[Tuple[int, ...], int]]:
    """Build one inverse map child_tuple -> parent for every rule level."""
    rule_list = _rules_sequence_to_numpy_list(rules)
    inverse_maps: List[Dict[Tuple[int, ...], int]] = []
    for level_rules in rule_list:
        inv: Dict[Tuple[int, ...], int] = {}
        num_parents, num_rules = int(level_rules.shape[0]), int(level_rules.shape[1])
        for parent in range(num_parents):
            for rr in range(num_rules):
                inv[tuple(int(x) for x in level_rules[parent, rr].tolist())] = int(parent)
        inverse_maps.append(inv)
    return inverse_maps


def grammar_validity_by_level(
    sequences: np.ndarray | torch.Tensor,
    rules: Sequence[torch.Tensor | np.ndarray] | Dict[int, Any],
    *,
    s: int,
) -> Dict[str, np.ndarray | float]:
    """
    Check whether generated terminal strings are compatible with the RHM grammar.

    Returns validity up to each hierarchy level.  For level ell=1, every block of
    s leaves must be a valid bottom-level production.  For ell=2, the inferred
    level-1 symbols must themselves form valid productions, and so on.  At
    ell=L, the whole string must reduce to one valid root symbol.

    Output:
      valid_by_level:       bool array [N,L]
      valid_frac_by_level:  float array [L]
      error_frac_by_level:  1 - valid_frac_by_level
      full_valid_frac:      valid_frac_by_level[-1]
      full_error_frac:      1 - full_valid_frac
    """
    if torch.is_tensor(sequences):
        seq = sequences.detach().cpu().numpy()
    else:
        seq = np.asarray(sequences)
    seq = np.asarray(seq, dtype=np.int64)
    if seq.ndim != 2:
        raise ValueError(f"sequences must have shape [N,d], got {seq.shape}.")

    rule_list = _rules_sequence_to_numpy_list(rules)
    inverse_maps = _inverse_maps_from_rules(rule_list)
    L = len(rule_list)
    N, d = int(seq.shape[0]), int(seq.shape[1])
    if d != int(s) ** L:
        raise ValueError(f"Expected sequence length s**L={int(s) ** L}, got d={d}.")

    current = seq.copy()
    valid_by_level = np.zeros((N, L), dtype=bool)

    # Reduce from leaves to root.  rule_level = L-ell is the correct
    # top-to-bottom rule index for the ell-th bottom-up reduction.
    for ell in range(1, L + 1):
        rule_level = L - ell
        inv = inverse_maps[rule_level]
        if current.shape[1] % int(s) != 0:
            raise ValueError(
                f"Internal parse width {current.shape[1]} not divisible by s={s} at level ell={ell}."
            )
        num_groups = current.shape[1] // int(s)
        parents = np.full((N, num_groups), -1, dtype=np.int64)

        for n in range(N):
            for g in range(num_groups):
                child_tuple_arr = current[n, g * int(s) : (g + 1) * int(s)]
                if np.any(child_tuple_arr < 0):
                    continue
                parent = inv.get(tuple(int(x) for x in child_tuple_arr.tolist()))
                if parent is not None:
                    parents[n, g] = int(parent)

        valid_by_level[:, ell - 1] = np.all(parents >= 0, axis=1)
        current = parents

    valid_frac = valid_by_level.mean(axis=0).astype(np.float64)
    err_frac = 1.0 - valid_frac
    return {
        "valid_by_level": valid_by_level,
        "valid_frac_by_level": valid_frac,
        "error_frac_by_level": err_frac,
        "full_valid_frac": float(valid_frac[-1]) if L > 0 else float("nan"),
        "full_error_frac": float(err_frac[-1]) if L > 0 else float("nan"),
    }




# -----------------------------------------------------------------------------
# Evaluation measures
# -----------------------------------------------------------------------------


def _init_level_accumulators(L: int) -> Dict[str, np.ndarray]:
    return {
        "A_mass_sum": np.zeros(L, dtype=np.float64),
        "B_mass_sum": np.zeros(L, dtype=np.float64),
        "margin_sum": np.zeros(L, dtype=np.float64),
        "margin_neg_count": np.zeros(L, dtype=np.float64),
        "margin_pos_count": np.zeros(L, dtype=np.float64),
        "penalty_sum": np.zeros(L, dtype=np.float64),
        "penalty_all_sum": np.zeros(L, dtype=np.float64),
        "valid_count": np.zeros(L, dtype=np.float64),
    }


def _update_hierarchy_accumulators(
    acc: Dict[str, np.ndarray],
    posterior: np.ndarray,
    true_tokens: np.ndarray,
    A_masks: np.ndarray,
    B_masks: np.ndarray,
    valid_masks: np.ndarray,
) -> Tuple[float, float]:
    """Update level accumulators. Returns residual_sum and reconstruction_abs_error_sum."""
    T, L, q = A_masks.shape
    residual_sum = 0.0
    recon_err_sum = 0.0
    for t in range(T):
        penalties_for_recon: List[float] = []
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
                acc["margin_neg_count"][ell] += float(margin < 0.0)
                acc["margin_pos_count"][ell] += float(margin > 0.0)
                acc["penalty_sum"][ell] += penalty
                acc["penalty_all_sum"][ell] += penalty
                acc["valid_count"][ell] += 1.0
                penalties_for_recon.append(penalty)
            else:
                acc["penalty_all_sum"][ell] += 0.0
                penalties_for_recon.append(0.0)

        A_last = A_masks[t, -1]
        pAL = float(posterior[t, A_last].sum()) if A_last.any() else 0.0
        p_true = float(posterior[t, int(true_tokens[t])])
        residual = float(np.log(max(pAL, EPS)) - np.log(max(p_true, EPS)))
        residual_sum += residual
        reconstructed = residual + float(np.sum(penalties_for_recon))
        actual_loss = float(-np.log(max(p_true, EPS)))
        recon_err_sum += abs(reconstructed - actual_loss)
    return residual_sum, recon_err_sum


@torch.no_grad()
def evaluate_diffusion_tasks_for_lambda(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    target_data: Dict[str, np.ndarray],
    *,
    lambda_radius: float,
    q: int,
    s: int,
    num_classes: int,
    batch_size: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    obs = observations
    N_obs = int(obs.shape[0])
    L = len(rules)

    flat_obs_index = target_data["flat_obs_index"]
    flat_target_pos = target_data["flat_target_pos"]
    flat_true_tokens = target_data["flat_true_tokens"]
    A_masks_all = target_data["A_masks"]
    B_masks_all = target_data["B_masks"]
    valid_masks_all = target_data["valid_masks"]
    T = int(flat_true_tokens.shape[0])

    losses: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    posterior_norms: List[np.ndarray] = []
    message_costs: List[np.ndarray] = []
    message_l2s: List[np.ndarray] = []
    max_msg_norms: List[np.ndarray] = []
    clipped_fracs: List[np.ndarray] = []

    acc = _init_level_accumulators(L)
    residual_sum = 0.0
    recon_err_sum = 0.0

    for start in range(0, N_obs, batch_size):
        end = min(start + batch_size, N_obs)
        obs_b = obs[start:end]
        res = bp_pass_local_torch(
            rules,
            rule_probs,
            obs_b,
            lambda_radius=float(lambda_radius),
            q=q,
            s=s,
            num_classes=num_classes,
            root_prior=root_prior,
            dtype=dtype,
        )

        message_costs.append(res.stats.total_cost.detach().cpu().numpy())
        message_l2s.append(res.stats.total_l2_norm.detach().cpu().numpy())
        max_msg_norms.append(res.stats.max_message_norm.detach().cpu().numpy())
        clipped_fracs.append(res.stats.clipped_fraction.detach().cpu().numpy())

        lo = int(np.searchsorted(flat_obs_index, start, side="left"))
        hi = int(np.searchsorted(flat_obs_index, end, side="left"))
        if hi <= lo:
            continue

        obs_local_np = flat_obs_index[lo:hi] - start
        pos_np = flat_target_pos[lo:hi]
        true_np = flat_true_tokens[lo:hi]
        obs_local = torch.from_numpy(obs_local_np).to(device=obs.device, dtype=torch.long)
        pos = torch.from_numpy(pos_np).to(device=obs.device, dtype=torch.long)
        true = torch.from_numpy(true_np).to(device=obs.device, dtype=torch.long)

        leaf_marg = res.marginals[-1]
        posterior = leaf_marg[obs_local, pos, :]
        pred = torch.argmax(posterior, dim=-1)
        p_true = posterior[torch.arange(posterior.shape[0], device=obs.device), true].clamp_min(EPS)
        loss = -torch.log(p_true)
        err = (pred != true).to(dtype)
        post_norm = centered_logit_l2_norm_torch(posterior)

        losses.append(loss.detach().cpu().numpy())
        errors.append(err.detach().cpu().numpy())
        posterior_norms.append(post_norm.detach().cpu().numpy())

        posterior_np = posterior.detach().cpu().numpy()
        rs, re = _update_hierarchy_accumulators(
            acc,
            posterior_np,
            true_np,
            A_masks_all[lo:hi],
            B_masks_all[lo:hi],
            valid_masks_all[lo:hi],
        )
        residual_sum += rs
        recon_err_sum += re

    losses_np = np.concatenate(losses) if losses else np.array([], dtype=np.float64)
    errors_np = np.concatenate(errors) if errors else np.array([], dtype=np.float64)
    posterior_norms_np = np.concatenate(posterior_norms) if posterior_norms else np.array([], dtype=np.float64)
    msg_cost_np = np.concatenate(message_costs)
    msg_l2_np = np.concatenate(message_l2s)
    max_msg_np = np.concatenate(max_msg_norms)
    clipped_np = np.concatenate(clipped_fracs)

    valid_count = acc["valid_count"]
    denom_valid = np.maximum(valid_count, 1.0)

    return {
        "lambda": float(lambda_radius),
        "loss_mean": float(losses_np.mean()),
        "loss_std": float(losses_np.std()),
        "error_mean": float(errors_np.mean()),
        "error_std": float(errors_np.std()),
        "posterior_norm_mean": float(posterior_norms_np.mean()),
        "posterior_norm_std": float(posterior_norms_np.std()),
        "message_total_cost_mean": float(msg_cost_np.mean()),
        "message_total_cost_std": float(msg_cost_np.std()),
        "message_total_l2_norm_mean": float(msg_l2_np.mean()),
        "message_total_l2_norm_std": float(msg_l2_np.std()),
        "message_max_norm_mean": float(max_msg_np.mean()),
        "message_max_norm_std": float(max_msg_np.std()),
        "message_clipped_fraction_mean": float(clipped_np.mean()),
        "message_clipped_fraction_std": float(clipped_np.std()),
        "num_observations": int(N_obs),
        "num_targets": int(T),
        "num_penalized_messages": int(res.stats.num_penalized_messages),
        "A_mass_mean": acc["A_mass_sum"] / denom_valid,
        "B_mass_mean": acc["B_mass_sum"] / denom_valid,
        "margin_mean": acc["margin_sum"] / denom_valid,
        "margin_neg_frac": acc["margin_neg_count"] / denom_valid,
        "margin_pos_frac": acc["margin_pos_count"] / denom_valid,
        "level_penalty_mean": acc["penalty_sum"] / denom_valid,
        "level_penalty_all_mean": acc["penalty_all_sum"] / float(max(T, 1)),
        "valid_level_frac": valid_count / float(max(T, 1)),
        "residual_mean": float(residual_sum / float(max(T, 1))),
        "reconstructed_loss_abs_error_mean": float(recon_err_sum / float(max(T, 1))),
    }


def _parse_float_list(text: Optional[str], default: Sequence[float]) -> List[float]:
    if text is None:
        return [float(x) for x in default]
    out: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"inf", "+inf", "infty", "infinite"}:
            out.append(float("inf"))
        else:
            out.append(float(item))
    return out


def _parse_lambda_values_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.lambda_values is not None:
        return np.array(_parse_float_list(args.lambda_values, []), dtype=np.float64)
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


def simulate_local_bp_diffusion_sweep(
    num_features: int = 16,
    num_classes: int = 16,
    num_synonyms: int = 4,
    tuple_size: int = 2,
    num_layers: int = 3,
    train_size: int = 1024,
    test_size: int = 1024,
    lambda_values: Optional[Sequence[float]] = None,
    seed_rules: int = 0,
    seed_sample: int = 0,
    seed_noise: int = 0,
    mask_prob_values: Sequence[float] = (1.0,),
    num_corruptions_per_sequence: int = 1,
    target_mode: str = "masked_only",
    margin_context: str = "clean",
    max_test_sequences: Optional[int] = None,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Notebook-friendly main function. Returns arrays indexed by lambda_values."""
    dev = _as_device(device)
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-2, 1.5, 20), [np.inf]))
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

    diff = make_masked_diffusion_observations(
        test_sequences,
        q=num_features,
        mask_prob_values=mask_prob_values,
        num_corruptions_per_sequence=num_corruptions_per_sequence,
        target_mode=target_mode,
        seed=seed_noise,
    )

    target_data = precompute_hierarchy_masks_for_diffusion_targets(
        diff["observations"],
        diff["clean_sequences"],
        diff["target_mask"],
        data["rules"],
        s=tuple_size,
        q=num_features,
        L=num_layers,
        margin_context=margin_context,
    )

    sweep: List[Dict[str, Any]] = []
    iterator = tqdm(lambda_values, desc="local BP diffusion sweep")
    for lam in iterator:
        out = evaluate_diffusion_tasks_for_lambda(
            data["rules"],
            data["rule_probs"],
            diff["observations"],
            target_data,
            lambda_radius=float(lam),
            q=num_features,
            s=tuple_size,
            num_classes=num_classes,
            batch_size=batch_size,
            dtype=dtype,
        )
        sweep.append(out)
        iterator.set_postfix(loss=f"{out['loss_mean']:.4g}")

    result: Dict[str, Any] = {
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
            "seed_noise": int(seed_noise),
            "mask_prob_values": [float(x) for x in mask_prob_values],
            "num_corruptions_per_sequence": int(num_corruptions_per_sequence),
            "target_mode": str(target_mode),
            "margin_context": str(margin_context),
            "max_test_sequences": None if max_test_sequences is None else int(max_test_sequences),
            "zipf": None if zipf is None else float(zipf),
            "layer": None if layer is None else int(layer),
            "replacement": None if replacement is None else bool(replacement),
            "last_layer_powerlaw_a": None if last_layer_powerlaw_a is None else float(last_layer_powerlaw_a),
            "batch_size": int(batch_size),
            "device": str(dev),
            "dtype": str(dtype).replace("torch.", ""),
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
        "message_max_norm_mean": _stack_sweep_array(sweep, "message_max_norm_mean"),
        "message_max_norm_std": _stack_sweep_array(sweep, "message_max_norm_std"),
        "message_clipped_fraction_mean": _stack_sweep_array(sweep, "message_clipped_fraction_mean"),
        "message_clipped_fraction_std": _stack_sweep_array(sweep, "message_clipped_fraction_std"),
        "A_mass_mean": _stack_sweep_array(sweep, "A_mass_mean"),
        "B_mass_mean": _stack_sweep_array(sweep, "B_mass_mean"),
        "margin_mean": _stack_sweep_array(sweep, "margin_mean"),
        "margin_neg_frac": _stack_sweep_array(sweep, "margin_neg_frac"),
        "margin_pos_frac": _stack_sweep_array(sweep, "margin_pos_frac"),
        "level_penalty_mean": _stack_sweep_array(sweep, "level_penalty_mean"),
        "level_penalty_all_mean": _stack_sweep_array(sweep, "level_penalty_all_mean"),
        "valid_level_frac": _stack_sweep_array(sweep, "valid_level_frac"),
        "residual_mean": _stack_sweep_array(sweep, "residual_mean"),
        "reconstructed_loss_abs_error_mean": _stack_sweep_array(sweep, "reconstructed_loss_abs_error_mean"),
        "num_observations": _stack_sweep_array(sweep, "num_observations"),
        "num_targets": _stack_sweep_array(sweep, "num_targets"),
        "num_penalized_messages": _stack_sweep_array(sweep, "num_penalized_messages"),
        "test_sequences": test_sequences.detach().cpu().numpy().astype(np.int64, copy=False),
        "observations": diff["observations"].detach().cpu().numpy().astype(np.int64, copy=False),
        "clean_sequences": diff["clean_sequences"].detach().cpu().numpy().astype(np.int64, copy=False),
        "target_mask": diff["target_mask"].detach().cpu().numpy().astype(bool, copy=False),
        "mask_prob_per_observation": diff["mask_prob_per_observation"].detach().cpu().numpy().astype(np.float64, copy=False),
        "flat_obs_index": target_data["flat_obs_index"],
        "flat_target_pos": target_data["flat_target_pos"],
        "flat_true_tokens": target_data["flat_true_tokens"],
        "A_masks": target_data["A_masks"],
        "B_masks": target_data["B_masks"],
        "valid_masks": target_data["valid_masks"],
        "raw_per_lambda": sweep,
        "note": (
            "Local constrained BP for masked diffusion on the RHM. lambda_values are per-message centered-logit L2 radii. "
            "The reported loss is the denoising cross-entropy averaged over generated corrupted observations and scored targets. "
            "With mask_prob_values=[1.0], this is the full-noise masked-diffusion slice. "
            "margin_context=clean defines the M_l partitions using the clean sequence, so M_l remains a level-wise evaluator even when x_t is full noise. "
            "level_penalty_mean is <log(1+exp(-M_l))> over valid level targets; margin_neg_frac is Pr[M_l<0]."
        ),
    }
    return result


def save_results_npz(results: Dict[str, Any], out_prefix: str | Path) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
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
        message_max_norm_mean=results["message_max_norm_mean"],
        message_max_norm_std=results["message_max_norm_std"],
        message_clipped_fraction_mean=results["message_clipped_fraction_mean"],
        message_clipped_fraction_std=results["message_clipped_fraction_std"],
        A_mass_mean=results["A_mass_mean"],
        B_mass_mean=results["B_mass_mean"],
        margin_mean=results["margin_mean"],
        margin_neg_frac=results["margin_neg_frac"],
        margin_pos_frac=results["margin_pos_frac"],
        level_penalty_mean=results["level_penalty_mean"],
        level_penalty_all_mean=results["level_penalty_all_mean"],
        valid_level_frac=results["valid_level_frac"],
        residual_mean=results["residual_mean"],
        reconstructed_loss_abs_error_mean=results["reconstructed_loss_abs_error_mean"],
        num_observations=results["num_observations"],
        num_targets=results["num_targets"],
        num_penalized_messages=results["num_penalized_messages"],
        test_sequences=results["test_sequences"],
        observations=results["observations"],
        clean_sequences=results["clean_sequences"],
        target_mask=results["target_mask"],
        mask_prob_per_observation=results["mask_prob_per_observation"],
        flat_obs_index=results["flat_obs_index"],
        flat_target_pos=results["flat_target_pos"],
        flat_true_tokens=results["flat_true_tokens"],
        A_masks=results["A_masks"],
        B_masks=results["B_masks"],
        valid_masks=results["valid_masks"],
        params_json=json.dumps(results["params"], sort_keys=True),
        note=results["note"],
    )
    return path


def load_results(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    if "params_json" in out:
        out["params"] = json.loads(str(out["params_json"]))
    return out


def _dtype_from_string(name: str) -> torch.dtype:
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError("dtype must be float64 or float32.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local constrained BP sweep for masked-diffusion denoising on the RHM.")
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--num_synonyms", type=int, default=4)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=1024)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--seed_noise", type=int, default=0)
    parser.add_argument("--mask_prob_values", type=str, default="1.0", help="Comma-separated mask probabilities; default 1.0 is full noise.")
    parser.add_argument("--num_corruptions_per_sequence", type=int, default=1)
    parser.add_argument("--target_mode", type=str, default="masked_only", choices=["masked_only", "all"])
    parser.add_argument("--margin_context", type=str, default="clean", choices=["clean", "observed"], help="A/B partitions for M_l: clean uses the full clean sequence as evaluator context; observed uses only x_t evidence.")
    parser.add_argument("--max_test_sequences", type=int, default=None)
    parser.add_argument("--zipf", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None)
    parser.add_argument("--lambda_values", type=str, default=None, help="Comma-separated local radii. Use 'inf' for exact BP.")
    parser.add_argument("--lambda_log10_min", type=float, default=-2.0)
    parser.add_argument("--lambda_log10_max", type=float, default=1.5)
    parser.add_argument("--lambda_num", type=int, default=20)
    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--include_inf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/bp_diffusion_local")
    args = parser.parse_args()

    lambda_values = _parse_lambda_values_from_args(args)
    mask_prob_values = _parse_float_list(args.mask_prob_values, [1.0])

    results = simulate_local_bp_diffusion_sweep(
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
        seed_noise=args.seed_noise,
        mask_prob_values=mask_prob_values,
        num_corruptions_per_sequence=args.num_corruptions_per_sequence,
        target_mode=args.target_mode,
        margin_context=args.margin_context,
        max_test_sequences=args.max_test_sequences,
        zipf=args.zipf,
        layer=args.layer,
        replacement=True if args.replacement else None,
        last_layer_powerlaw_a=args.last_layer_powerlaw_a,
        batch_size=args.batch_size,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
    )

    path = save_results_npz(results, args.out_prefix)
    print("Saved", path)
    print(results["note"])
    print("params:", json.dumps(results["params"], indent=2, sort_keys=True))
    print("lambda_values:", results["lambda_values"])
    print("loss_mean:", results["loss_mean"])
    print("error_mean:", results["error_mean"])
    print("margin_neg_frac:\n", results["margin_neg_frac"])
    print("level_penalty_mean:\n", results["level_penalty_mean"])
    print("valid_level_frac:\n", results["valid_level_frac"])
    print("reconstructed_loss_abs_error_mean:", results["reconstructed_loss_abs_error_mean"])


if __name__ == "__main__":
    main()
