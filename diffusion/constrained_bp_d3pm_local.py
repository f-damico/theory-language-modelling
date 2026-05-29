#!/usr/bin/env python3
"""
Local-norm constrained oracle BP used as the denoiser inside a D3PM-style
uniform discrete diffusion model on the Random Hierarchy Model (RHM).

This is the object that is closest to the RHM diffusion experiments in the
papers using D3PMs: the neural network's prediction p_theta(x0 | x_t, t) is
replaced by the constrained-BP posterior p_lambda(x0 | x_t, t), and generation
uses the same D3PM reverse posterior formula.

Two outputs are computed for a grid of local message radii lambda:

1) Denoising test CE, averaged over clean test sequences, sampled diffusion
   times, and forward-corrupted x_t:

       L_CE(lambda) = E[- 1/d sum_i log p_lambda(x0_i | x_t, t)].

   This is the clean-token x0-prediction loss. It is not the full D3PM ELBO,
   but it is the loss directly decomposable into RHM level margins.

2) Generated-sample grammar validity, obtained by starting from uniform noise
   x_T, running the D3PM reverse chain with constrained BP, and checking whether
   final samples are compatible with the RHM grammar at each level.

Notebook usage
--------------
    import numpy as np
    import constrained_bp_d3pm_local as bpd

    res = bpd.simulate_local_bp_d3pm_sweep(
        num_features=16, num_classes=16, num_synonyms=4,
        tuple_size=2, num_layers=3,
        test_size=512,
        lambda_values=np.concatenate(([0.0], np.logspace(-2, 2, 20), [np.inf])),
        diffusion_steps=100,
        num_time_samples_per_sequence=1,
        compute_generation=True,
        num_generated=1024,
        device="cuda",
    )
    bpd.save_results_npz(res, "./bp_d3pm_local_L3")

Terminal usage
--------------
    python constrained_bp_d3pm_local.py \
        --num_features 16 --num_classes 16 --num_synonyms 4 \
        --tuple_size 2 --num_layers 3 --test_size 1024 \
        --diffusion_steps 100 --num_time_samples_per_sequence 1 \
        --include_zero --include_inf --lambda_log10_min -2 --lambda_log10_max 2 --lambda_num 25 \
        --compute_generation --num_generated 1024 \
        --device cuda --out_prefix ./data/BP/bp_d3pm_local_L3
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


# -----------------------------------------------------------------------------
# Standalone local-BP/RHM utilities copied here intentionally.
# Do not import constrained_bp_diffusion_local: cluster copies can be stale.
# -----------------------------------------------------------------------------

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
# Uniform D3PM schedule and kernels
# -----------------------------------------------------------------------------


def make_beta_schedule(
    diffusion_steps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "linear",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """
    Create one-step beta_t and cumulative alpha_bar_t for t=0,...,T.

    The uniform transition is

        Q_t = alpha_t I + (1 - alpha_t) 11^T / q,

    where alpha_t = 1 - beta_t.  The cumulative transition is

        \bar Q_t = alpha_bar_t I + (1 - alpha_bar_t) 11^T / q.

    alpha_bar[0] = 1, alpha_bar[t] = prod_{r=1}^t alpha_r.
    """
    T = int(diffusion_steps)
    if T <= 0:
        raise ValueError("diffusion_steps must be positive.")
    dev = _as_device(device)
    if schedule == "linear":
        beta = torch.linspace(float(beta_start), float(beta_end), T, dtype=dtype, device=dev)
    elif schedule == "constant":
        beta = torch.full((T,), float(beta_end), dtype=dtype, device=dev)
    else:
        raise ValueError("schedule must be 'linear' or 'constant'.")
    beta = beta.clamp(min=1e-12, max=1.0 - 1e-12)
    alpha = 1.0 - beta
    alpha_bar = torch.empty((T + 1,), dtype=dtype, device=dev)
    alpha_bar[0] = 1.0
    alpha_bar[1:] = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar}


def d3pm_uniform_likelihood_from_observation(
    x_t: torch.Tensor,
    t_index: torch.Tensor,
    *,
    q: int,
    alpha_bar: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Leaf evidence psi_i(a) = q(x_{t,i} | x0_i=a) for the uniform D3PM kernel.

    x_t has shape [B,d] and t_index has shape [B] with values in {1,...,T}.
    Returns evidence [B,d,q].
    """
    x_t = x_t.to(torch.long)
    B, d = x_t.shape
    dev = x_t.device
    ab = alpha_bar[t_index.to(torch.long)].to(device=dev, dtype=dtype).reshape(B, 1, 1)
    base = (1.0 - ab) / float(q)
    evidence = base.expand(B, d, q).clone()
    row = torch.arange(B, device=dev).reshape(B, 1).expand(B, d)
    col = torch.arange(d, device=dev).reshape(1, d).expand(B, d)
    evidence[row, col, x_t] += ab.squeeze(-1)
    return evidence


def sample_d3pm_forward_uniform(
    x0: torch.Tensor,
    t_index: torch.Tensor,
    *,
    q: int,
    alpha_bar: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x0) for the cumulative uniform D3PM kernel."""
    x0 = x0.to(torch.long)
    B, d = x0.shape
    dev = x0.device
    ab = alpha_bar[t_index.to(torch.long)].to(device=dev, dtype=torch.float64).reshape(B, 1)
    keep = torch.rand((B, d), device=dev, generator=generator, dtype=torch.float64) < ab
    noise = torch.randint(0, int(q), size=(B, d), device=dev, generator=generator, dtype=torch.long)
    return torch.where(keep, x0, noise)


def d3pm_uniform_reverse_from_x0_posterior(
    p0: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    *,
    q: int,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute p(x_{t-1}=b | x_t=c) by mixing the exact posterior
    q(x_{t-1}=b | x_t=c, x0=a) over p0(a) = p_lambda(x0=a | x_t).

    p0: [B,d,q]
    x_t: [B,d]
    t: integer in {1,...,T}
    returns: [B,d,q]
    """
    if t < 1:
        raise ValueError("t must be >= 1.")
    dev = p0.device
    dtype = p0.dtype
    B, d, q_here = p0.shape
    if q_here != int(q):
        raise ValueError("p0 last dimension must equal q.")

    a_step = alpha[int(t) - 1].to(device=dev, dtype=dtype)
    ab_t = alpha_bar[int(t)].to(device=dev, dtype=dtype)
    ab_prev = alpha_bar[int(t) - 1].to(device=dev, dtype=dtype)

    states = torch.arange(int(q), dtype=torch.long, device=dev)
    c = x_t.to(torch.long).unsqueeze(-1)  # [B,d,1]

    # denom[a] = q(x_t=c | x0=a) = ab_t * 1[c=a] + (1-ab_t)/q.
    denom = (1.0 - ab_t) / float(q) + ab_t * (states.reshape(1, 1, q) == c).to(dtype)
    weighted = p0 / denom.clamp_min(1e-30)  # [B,d,a]

    # For each previous-token b, compute:
    # sum_a p0[a] * qbar_{t-1}(b|a) / qbar_t(c|a), then multiply by Q_t(c|b).
    out = torch.empty_like(p0)
    for b in range(int(q)):
        qprev_b_given_a = (1.0 - ab_prev) / float(q) + ab_prev * (states == b).to(dtype)  # [a]
        mix_b = (weighted * qprev_b_given_a.reshape(1, 1, q)).sum(dim=-1)  # [B,d]
        qstep_c_given_b = (1.0 - a_step) / float(q) + a_step * (x_t == b).to(dtype)  # [B,d]
        out[:, :, b] = qstep_c_given_b * mix_b

    return normalize(out, dim=-1)


# -----------------------------------------------------------------------------
# BP with soft D3PM evidence
# -----------------------------------------------------------------------------


@torch.no_grad()
def bp_pass_local_evidence_torch(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    leaf_evidence: torch.Tensor,
    *,
    lambda_radius: float,
    q: int,
    s: int,
    num_classes: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> BPBatchResultLocal:
    """Run local-constrained BP with arbitrary positive leaf evidence [B,d,q]."""
    evidence = normalize(leaf_evidence.to(dtype=dtype), dim=-1)
    dev = evidence.device
    L = len(rules)
    B, d, q_e = evidence.shape
    if q_e != int(q):
        raise ValueError("leaf_evidence last dimension must equal q.")
    if d != int(s) ** L:
        raise ValueError(f"sequence length {d} must equal s**L={int(s)**L}.")

    subtree: List[torch.Tensor] = []
    context: List[torch.Tensor] = []
    for depth in range(L + 1):
        K = node_state_dim(depth, q=q, num_classes=num_classes)
        n_nodes = int(s) ** depth
        subtree.append(torch.full((B, n_nodes, K), 1.0 / K, dtype=dtype, device=dev))
        context.append(torch.full((B, n_nodes, K), 1.0 / K, dtype=dtype, device=dev))
    subtree[L] = evidence

    total_cost = torch.zeros(B, dtype=dtype, device=dev)
    total_l2 = torch.zeros(B, dtype=dtype, device=dev)
    max_norm = torch.zeros(B, dtype=dtype, device=dev)
    clipped_count = torch.zeros(B, dtype=dtype, device=dev)
    n_penalized = 0

    # Upward pass.
    for depth in range(L - 1, -1, -1):
        n_nodes = int(s) ** depth
        child_block = subtree[depth + 1].reshape(B, n_nodes, int(s), -1)
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
        n_nodes = int(s) ** depth
        children_subtree = subtree[depth + 1].reshape(B, n_nodes, int(s), -1)
        out_children = torch.empty_like(children_subtree)
        for child_pos in range(int(s)):
            cand = downward_candidate_for_child(
                context[depth], children_subtree, rules[depth], rule_probs[depth], child_pos
            )
            msg, cost, l2, mx, clipped = project_local_centered_logit_ball_batch(cand, lambda_radius)
            out_children[:, :, child_pos, :] = msg
            total_cost += cost
            total_l2 += l2
            max_norm = torch.maximum(max_norm, mx)
            clipped_count += clipped
            n_penalized += n_nodes
        context[depth + 1] = out_children.reshape(B, n_nodes * int(s), -1)

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
# Denoising CE evaluation and M_l decomposition
# -----------------------------------------------------------------------------


def make_d3pm_test_tasks(
    sequences: torch.Tensor,
    *,
    q: int,
    alpha_bar: torch.Tensor,
    num_time_samples_per_sequence: int = 1,
    time_sampling: str = "uniform",
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Sample t and x_t for denoising-test evaluation."""
    if time_sampling != "uniform":
        raise ValueError("Only uniform time_sampling is implemented.")
    seq = sequences.to(torch.long)
    dev = seq.device
    N, d = seq.shape
    T = int(alpha_bar.numel()) - 1
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    clean_list: List[torch.Tensor] = []
    xt_list: List[torch.Tensor] = []
    t_list: List[torch.Tensor] = []
    target_mask_list: List[torch.Tensor] = []

    for _ in range(int(num_time_samples_per_sequence)):
        t_index = torch.randint(1, T + 1, size=(N,), dtype=torch.long, device=dev, generator=gen)
        x_t = sample_d3pm_forward_uniform(seq, t_index, q=q, alpha_bar=alpha_bar, generator=gen)
        clean_list.append(seq.clone())
        xt_list.append(x_t)
        t_list.append(t_index)
        target_mask_list.append(torch.ones((N, d), dtype=torch.bool, device=dev))

    return {
        "clean_sequences": torch.cat(clean_list, dim=0),
        "x_t": torch.cat(xt_list, dim=0),
        "t_index": torch.cat(t_list, dim=0),
        "target_mask": torch.cat(target_mask_list, dim=0),
    }


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
    T, L, q = A_masks.shape
    residual_sum = 0.0
    recon_err_sum = 0.0

    for idx in range(T):
        p = np.asarray(posterior[idx], dtype=np.float64)
        y = int(true_tokens[idx])
        py = float(max(p[y], EPS))
        level_penalty_sum = 0.0
        last_A_mass = py

        for ell in range(L):
            A = A_masks[idx, ell]
            B = B_masks[idx, ell]
            valid = bool(valid_masks[idx, ell]) and A.any() and B.any()
            if not valid:
                # This level contributes zero to the exact A/B telescoping; keep
                # level_penalty_all_mean finite and count no valid diagnostic.
                continue
            PA = float(max(p[A].sum(), EPS))
            PB = float(max(p[B].sum(), EPS))
            M = math.log(PA) - math.log(PB)
            penalty = math.log1p(math.exp(-M))
            level_penalty_sum += penalty
            last_A_mass = PA

            acc["A_mass_sum"][ell] += PA
            acc["B_mass_sum"][ell] += PB
            acc["margin_sum"][ell] += M
            acc["margin_neg_count"][ell] += 1.0 if M < 0.0 else 0.0
            acc["margin_pos_count"][ell] += 1.0 if M > 0.0 else 0.0
            acc["penalty_sum"][ell] += penalty
            acc["penalty_all_sum"][ell] += penalty
            acc["valid_count"][ell] += 1.0

        residual = -math.log(py / max(last_A_mass, EPS))
        residual_sum += residual
        recon = residual + level_penalty_sum
        true_loss = -math.log(py)
        recon_err_sum += abs(recon - true_loss)

    return residual_sum, recon_err_sum


@torch.no_grad()
def evaluate_d3pm_ce_one_lambda(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    tasks: Dict[str, torch.Tensor],
    target_data: Dict[str, np.ndarray],
    *,
    lambda_radius: float,
    q: int,
    s: int,
    num_classes: int,
    alpha_bar: torch.Tensor,
    root_prior: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Evaluate clean-token CE and level-wise M_l decomposition for one lambda."""
    x_t = tasks["x_t"]
    t_index = tasks["t_index"]
    clean = tasks["clean_sequences"]
    target_mask = tasks["target_mask"]
    dev = x_t.device
    B_total, d = x_t.shape
    L = len(rules)

    # The A/B masks are flat over all positions of all observations.
    flat_obs = target_data["flat_obs_index"]
    flat_pos = target_data["flat_target_pos"]
    flat_true = target_data["flat_true_tokens"]
    A_masks = target_data["A_masks"]
    B_masks = target_data["B_masks"]
    valid_masks = target_data["valid_masks"]

    losses: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    posterior_norms: List[np.ndarray] = []
    msg_costs: List[np.ndarray] = []
    msg_l2s: List[np.ndarray] = []
    msg_maxs: List[np.ndarray] = []
    clipped_fracs: List[np.ndarray] = []
    acc = _init_level_accumulators(L)
    residual_sum_total = 0.0
    recon_err_sum_total = 0.0

    # For fast lookup of flat target indices by observation interval, flat_obs is
    # produced in row-major order by precompute_hierarchy_masks_for_diffusion_targets.
    for start in range(0, B_total, int(batch_size)):
        end = min(start + int(batch_size), B_total)
        xb = x_t[start:end]
        tb = t_index[start:end]
        evidence = d3pm_uniform_likelihood_from_observation(xb, tb, q=q, alpha_bar=alpha_bar, dtype=dtype)
        res = bp_pass_local_evidence_torch(
            rules,
            rule_probs,
            evidence,
            lambda_radius=float(lambda_radius),
            q=q,
            s=s,
            num_classes=num_classes,
            root_prior=root_prior,
            dtype=dtype,
        )
        leaf_post = res.marginals[-1]  # [B,d,q]
        yb = clean[start:end]
        probs_true = leaf_post.gather(2, yb.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
        batch_loss = -torch.log(probs_true)
        pred = leaf_post.argmax(dim=-1)
        batch_err = (pred != yb).to(dtype)
        batch_norm = centered_logit_l2_norm_torch(leaf_post.reshape(-1, q)).reshape(end - start, d)

        losses.append(batch_loss.reshape(-1).detach().cpu().numpy())
        errors.append(batch_err.reshape(-1).detach().cpu().numpy())
        posterior_norms.append(batch_norm.reshape(-1).detach().cpu().numpy())
        msg_costs.append(res.stats.total_cost.detach().cpu().numpy())
        msg_l2s.append(res.stats.total_l2_norm.detach().cpu().numpy())
        msg_maxs.append(res.stats.max_message_norm.detach().cpu().numpy())
        clipped_fracs.append(res.stats.clipped_fraction.detach().cpu().numpy())

        # Extract all flat targets in this observation batch. Since target_mask is
        # all ones, global flat index = obs_index*d + pos, while posterior_flat
        # below is local to the current batch.
        lo = start * d
        hi = end * d
        flat_slice = slice(lo, hi)
        posterior_flat = leaf_post.reshape(-1, q).detach().cpu().numpy()
        true_sel = yb.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
        residual_sum, recon_err_sum = _update_hierarchy_accumulators(
            acc,
            posterior_flat,
            true_sel,
            A_masks[flat_slice],
            B_masks[flat_slice],
            valid_masks[flat_slice],
        )
        residual_sum_total += residual_sum
        recon_err_sum_total += recon_err_sum

    losses_np = np.concatenate(losses)
    errors_np = np.concatenate(errors)
    posterior_norms_np = np.concatenate(posterior_norms)
    msg_cost_np = np.concatenate(msg_costs)
    msg_l2_np = np.concatenate(msg_l2s)
    msg_max_np = np.concatenate(msg_maxs)
    clipped_np = np.concatenate(clipped_fracs)
    valid_count = acc["valid_count"]
    denom_valid = np.maximum(valid_count, 1.0)
    T_targets = int(losses_np.size)

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
        "message_max_norm_mean": float(msg_max_np.mean()),
        "message_max_norm_std": float(msg_max_np.std()),
        "message_clipped_fraction_mean": float(clipped_np.mean()),
        "message_clipped_fraction_std": float(clipped_np.std()),
        "num_observations": int(B_total),
        "num_targets": int(T_targets),
        "num_penalized_messages": int(res.stats.num_penalized_messages),
        "A_mass_mean": acc["A_mass_sum"] / denom_valid,
        "B_mass_mean": acc["B_mass_sum"] / denom_valid,
        "margin_mean": acc["margin_sum"] / denom_valid,
        "margin_neg_frac": acc["margin_neg_count"] / denom_valid,
        "margin_pos_frac": acc["margin_pos_count"] / denom_valid,
        "level_penalty_mean": acc["penalty_sum"] / denom_valid,
        "level_penalty_all_mean": acc["penalty_all_sum"] / float(max(T_targets, 1)),
        "valid_level_frac": valid_count / float(max(T_targets, 1)),
        "residual_mean": float(residual_sum_total / float(max(T_targets, 1))),
        "reconstructed_loss_abs_error_mean": float(recon_err_sum_total / float(max(T_targets, 1))),
    }


# -----------------------------------------------------------------------------
# D3PM generation using constrained BP as x0-predictor
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate_samples_with_local_bp_d3pm(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    *,
    lambda_radius: float,
    q: int,
    s: int,
    num_classes: int,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    num_samples: int = 1024,
    batch_size: int = 128,
    seed: int = 123,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    save_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Start from x_T ~ uniform and run the D3PM reverse chain with constrained BP.

    This is the closest BP analogue of a D3PM network sampler: BP supplies
    p_lambda(x0 | x_t,t), then the exact D3PM reverse posterior formula maps it
    to p_lambda(x_{t-1} | x_t,t).
    """
    dev = rules[0].device
    L = len(rules)
    d = int(s) ** L
    T_steps = int(alpha.numel())
    N = int(num_samples)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    generated_chunks: List[torch.Tensor] = []
    msg_cost_chunks: List[np.ndarray] = []
    msg_l2_chunks: List[np.ndarray] = []
    clip_chunks: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []

    for start in tqdm(range(0, N, int(batch_size)), desc=f"generate lambda={lambda_radius}", leave=False):
        B = min(int(batch_size), N - start)
        x_t = torch.randint(0, int(q), size=(B, d), dtype=torch.long, device=dev, generator=gen)
        if save_trajectory:
            traj = [x_t.detach().cpu().numpy().astype(np.int64, copy=False)]

        last_stats = None
        for t in range(T_steps, 0, -1):
            t_batch = torch.full((B,), int(t), dtype=torch.long, device=dev)
            evidence = d3pm_uniform_likelihood_from_observation(x_t, t_batch, q=q, alpha_bar=alpha_bar, dtype=dtype)
            res = bp_pass_local_evidence_torch(
                rules,
                rule_probs,
                evidence,
                lambda_radius=float(lambda_radius),
                q=q,
                s=s,
                num_classes=num_classes,
                root_prior=root_prior,
                dtype=dtype,
            )
            p0 = res.marginals[-1]
            p_prev = d3pm_uniform_reverse_from_x0_posterior(
                p0,
                x_t,
                t,
                q=q,
                alpha=alpha,
                alpha_bar=alpha_bar,
            )
            x_t = torch.multinomial(p_prev.reshape(-1, int(q)), num_samples=1, replacement=True, generator=gen).reshape(B, d)
            last_stats = res.stats
            if save_trajectory:
                traj.append(x_t.detach().cpu().numpy().astype(np.int64, copy=False))

        generated_chunks.append(x_t.detach().cpu())
        if last_stats is not None:
            msg_cost_chunks.append(last_stats.total_cost.detach().cpu().numpy())
            msg_l2_chunks.append(last_stats.total_l2_norm.detach().cpu().numpy())
            clip_chunks.append(last_stats.clipped_fraction.detach().cpu().numpy())
        if save_trajectory:
            trajectories.append(np.stack(traj, axis=1))  # [B,T+1,d]

    generated = torch.cat(generated_chunks, dim=0).numpy().astype(np.int64, copy=False)
    validity = grammar_validity_by_level(generated, rules, s=s)
    out: Dict[str, Any] = {
        "generated_sequences": generated,
        "generated_valid_by_level": validity["valid_by_level"],
        "generated_valid_frac_by_level": validity["valid_frac_by_level"],
        "generated_error_frac_by_level": validity["error_frac_by_level"],
        "generated_full_valid_frac": validity["full_valid_frac"],
        "generated_full_error_frac": validity["full_error_frac"],
        "generated_num_samples": int(N),
    }
    if msg_cost_chunks:
        out["generation_last_step_message_total_cost_mean"] = float(np.concatenate(msg_cost_chunks).mean())
        out["generation_last_step_message_total_l2_norm_mean"] = float(np.concatenate(msg_l2_chunks).mean())
        out["generation_last_step_message_clipped_fraction_mean"] = float(np.concatenate(clip_chunks).mean())
    if save_trajectory:
        out["generated_trajectory"] = np.concatenate(trajectories, axis=0)
    return out


# -----------------------------------------------------------------------------
# Sweep, saving, CLI
# -----------------------------------------------------------------------------


def _parse_float_list(text: Optional[str], default: Sequence[float]) -> List[float]:
    if text is None:
        return [float(x) for x in default]
    out: List[float] = []
    for item in str(text).split(","):
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


def simulate_local_bp_d3pm_sweep(
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
    diffusion_steps: int = 100,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    beta_schedule: str = "linear",
    num_time_samples_per_sequence: int = 1,
    margin_context: str = "clean",
    max_test_sequences: Optional[int] = None,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
    batch_size: int = 128,
    compute_generation: bool = True,
    num_generated: int = 1024,
    generation_batch_size: Optional[int] = None,
    generation_seed: int = 123,
    save_generated_samples: bool = False,
    save_generated_trajectory: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Notebook-friendly main sweep. Arrays are indexed by lambda_values."""
    dev = _as_device(device)
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-2, 2, 25), [np.inf]))
    lambda_values = np.asarray(lambda_values, dtype=np.float64)

    schedule = make_beta_schedule(
        diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=beta_schedule,
        device=dev,
        dtype=dtype,
    )
    alpha = schedule["alpha"]
    alpha_bar = schedule["alpha_bar"]

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

    tasks = make_d3pm_test_tasks(
        test_sequences,
        q=num_features,
        alpha_bar=alpha_bar,
        num_time_samples_per_sequence=num_time_samples_per_sequence,
        seed=seed_noise,
    )

    # For the margin decomposition, all positions are scored.  The clean context
    # gives the evaluator's nested RHM branches A_l/B_l for each true token.
    target_data = precompute_hierarchy_masks_for_diffusion_targets(
        tasks["x_t"],
        tasks["clean_sequences"],
        tasks["target_mask"],
        data["rules"],
        s=tuple_size,
        q=num_features,
        L=num_layers,
        margin_context=margin_context,
    )

    sweep: List[Dict[str, Any]] = []
    for lam in tqdm(lambda_values, desc="lambda sweep"):
        metrics = evaluate_d3pm_ce_one_lambda(
            data["rules"],
            data["rule_probs"],
            tasks,
            target_data,
            lambda_radius=float(lam),
            q=num_features,
            s=tuple_size,
            num_classes=num_classes,
            alpha_bar=alpha_bar,
            batch_size=batch_size,
            dtype=dtype,
        )
        if compute_generation:
            gen_metrics = generate_samples_with_local_bp_d3pm(
                data["rules"],
                data["rule_probs"],
                lambda_radius=float(lam),
                q=num_features,
                s=tuple_size,
                num_classes=num_classes,
                alpha=alpha,
                alpha_bar=alpha_bar,
                num_samples=num_generated,
                batch_size=generation_batch_size or batch_size,
                seed=generation_seed,
                dtype=dtype,
                save_trajectory=save_generated_trajectory,
            )
            for key, value in gen_metrics.items():
                if key in {"generated_sequences", "generated_valid_by_level", "generated_trajectory"} and not save_generated_samples and key != "generated_trajectory":
                    continue
                if key == "generated_trajectory" and not save_generated_trajectory:
                    continue
                metrics[key] = value
        sweep.append(metrics)

    params = {
        "num_features": num_features,
        "num_classes": num_classes,
        "num_synonyms": num_synonyms,
        "tuple_size": tuple_size,
        "num_layers": num_layers,
        "train_size": train_size,
        "test_size": test_size,
        "seed_rules": seed_rules,
        "seed_sample": seed_sample,
        "seed_noise": seed_noise,
        "diffusion_steps": diffusion_steps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "beta_schedule": beta_schedule,
        "alpha_bar_T": float(alpha_bar[-1].detach().cpu().item()),
        "num_time_samples_per_sequence": num_time_samples_per_sequence,
        "margin_context": margin_context,
        "zipf": zipf,
        "layer": layer,
        "replacement": replacement,
        "last_layer_powerlaw_a": last_layer_powerlaw_a,
        "batch_size": batch_size,
        "compute_generation": compute_generation,
        "num_generated": num_generated,
        "generation_seed": generation_seed,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(dev),
    }

    result: Dict[str, Any] = {
        "params": params,
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
        "alpha": alpha.detach().cpu().numpy(),
        "alpha_bar": alpha_bar.detach().cpu().numpy(),
        "test_sequences": test_sequences.detach().cpu().numpy().astype(np.int64, copy=False),
        "x_t": tasks["x_t"].detach().cpu().numpy().astype(np.int64, copy=False),
        "clean_sequences": tasks["clean_sequences"].detach().cpu().numpy().astype(np.int64, copy=False),
        "t_index": tasks["t_index"].detach().cpu().numpy().astype(np.int64, copy=False),
        "target_mask": tasks["target_mask"].detach().cpu().numpy().astype(bool, copy=False),
        "flat_obs_index": target_data["flat_obs_index"],
        "flat_target_pos": target_data["flat_target_pos"],
        "flat_true_tokens": target_data["flat_true_tokens"],
        "A_masks": target_data["A_masks"],
        "B_masks": target_data["B_masks"],
        "valid_masks": target_data["valid_masks"],
        "raw_per_lambda": sweep,
        "note": (
            "Local constrained BP with uniform-D3PM forward and reverse kernels on the RHM. "
            "BP is used as the x0-predictor p_lambda(x0|x_t,t), replacing the neural network denoiser. "
            "loss_mean is clean-token denoising CE averaged over sampled diffusion times and all positions. "
            "level_penalty_mean and margin_neg_frac give the exact RHM M_l decomposition of that CE. "
            "generated_error_frac_by_level / generated_valid_frac_by_level are obtained by running the full D3PM reverse chain from uniform noise."
        ),
    }
    if compute_generation:
        result["generated_valid_frac_by_level"] = _stack_sweep_array(sweep, "generated_valid_frac_by_level")
        result["generated_error_frac_by_level"] = _stack_sweep_array(sweep, "generated_error_frac_by_level")
        result["generated_full_valid_frac"] = _stack_sweep_array(sweep, "generated_full_valid_frac")
        result["generated_full_error_frac"] = _stack_sweep_array(sweep, "generated_full_error_frac")
        result["generated_num_samples"] = _stack_sweep_array(sweep, "generated_num_samples")
        for optional_key in ("generation_last_step_message_total_cost_mean", "generation_last_step_message_total_l2_norm_mean", "generation_last_step_message_clipped_fraction_mean"):
            if optional_key in sweep[0]:
                result[optional_key] = _stack_sweep_array(sweep, optional_key)
        if save_generated_samples and "generated_sequences" in sweep[0]:
            result["generated_sequences_by_lambda"] = _stack_sweep_array(sweep, "generated_sequences")
            result["generated_valid_by_level_by_lambda"] = _stack_sweep_array(sweep, "generated_valid_by_level")
        if save_generated_trajectory and "generated_trajectory" in sweep[0]:
            result["generated_trajectory_by_lambda"] = _stack_sweep_array(sweep, "generated_trajectory")

    return result


def save_results_npz(results: Dict[str, Any], out_prefix: str | Path) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = Path(str(out_prefix) + ".npz")
    extra_save: Dict[str, Any] = {}
    for key in (
        "generated_valid_frac_by_level",
        "generated_error_frac_by_level",
        "generated_full_valid_frac",
        "generated_full_error_frac",
        "generated_num_samples",
        "generation_last_step_message_total_cost_mean",
        "generation_last_step_message_total_l2_norm_mean",
        "generation_last_step_message_clipped_fraction_mean",
        "generated_sequences_by_lambda",
        "generated_valid_by_level_by_lambda",
        "generated_trajectory_by_lambda",
    ):
        if key in results:
            extra_save[key] = results[key]
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
        alpha=results["alpha"],
        alpha_bar=results["alpha_bar"],
        test_sequences=results["test_sequences"],
        x_t=results["x_t"],
        clean_sequences=results["clean_sequences"],
        t_index=results["t_index"],
        target_mask=results["target_mask"],
        flat_obs_index=results["flat_obs_index"],
        flat_target_pos=results["flat_target_pos"],
        flat_true_tokens=results["flat_true_tokens"],
        A_masks=results["A_masks"],
        B_masks=results["B_masks"],
        valid_masks=results["valid_masks"],
        params_json=json.dumps(results["params"], sort_keys=True),
        note=results["note"],
        **extra_save,
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
    parser = argparse.ArgumentParser(description="Local constrained BP as uniform-D3PM denoiser on the RHM.")
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
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "constant"])
    parser.add_argument("--num_time_samples_per_sequence", type=int, default=1)
    parser.add_argument("--margin_context", type=str, default="clean", choices=["clean", "observed"])
    parser.add_argument("--max_test_sequences", type=int, default=None)
    parser.add_argument("--zipf", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None)
    parser.add_argument("--lambda_values", type=str, default=None, help="Comma-separated local radii. Use 'inf' for exact BP.")
    parser.add_argument("--lambda_log10_min", type=float, default=-2.0)
    parser.add_argument("--lambda_log10_max", type=float, default=2.0)
    parser.add_argument("--lambda_num", type=int, default=25)
    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--include_inf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--compute_generation", action="store_true")
    parser.add_argument("--no_compute_generation", dest="compute_generation", action="store_false")
    parser.set_defaults(compute_generation=True)
    parser.add_argument("--num_generated", type=int, default=1024)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--generation_seed", type=int, default=123)
    parser.add_argument("--save_generated_samples", action="store_true")
    parser.add_argument("--save_generated_trajectory", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/bp_d3pm_local")
    args = parser.parse_args()

    lambda_values = _parse_lambda_values_from_args(args)
    results = simulate_local_bp_d3pm_sweep(
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
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        num_time_samples_per_sequence=args.num_time_samples_per_sequence,
        margin_context=args.margin_context,
        max_test_sequences=args.max_test_sequences,
        zipf=args.zipf,
        layer=args.layer,
        replacement=True if args.replacement else None,
        last_layer_powerlaw_a=args.last_layer_powerlaw_a,
        batch_size=args.batch_size,
        compute_generation=args.compute_generation,
        num_generated=args.num_generated,
        generation_batch_size=args.generation_batch_size,
        generation_seed=args.generation_seed,
        save_generated_samples=args.save_generated_samples,
        save_generated_trajectory=args.save_generated_trajectory,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
    )
    path = save_results_npz(results, args.out_prefix)
    print(f"[DONE] saved {path}")


if __name__ == "__main__":
    main()
