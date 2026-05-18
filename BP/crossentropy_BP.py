#!/usr/bin/env python3
"""
Cross-entropy / KL-style gradient descent in BP-message space for the RHM.

This file is meant to sit next to ``constrained_bp_last_next_torch.py`` and to
reuse its exact RHM sampler, BP tensor utilities, and result conventions.

What is different from the local/global budget BP files?
-------------------------------------------------------
The local/global files compute BP messages by running a BP recursion and then
clipping/shrinking centered log-message vectors at a prescribed budget.  Here we
instead make the internal BP messages themselves trainable variables:

    c_e  ->  m_e = softmax(c_e),      mean(c_e)=irrelevant.

The messages are initialized with a very small but non-zero centered-logit norm.
Then we minimize a cross-entropy objective for the selected prediction tasks.
At every evaluation step we measure the realized message-logit norm and the same
hierarchy observables used in the previous constrained-BP analysis:

    - loss and top-1 error;
    - total centered-message logit cost/norm;
    - posterior centered-logit norm;
    - A_l/B_l masses;
    - M_l margins;
    - peeled loss <log(1+exp(-M_l))>;
    - Pr(M_l>0) and Pr(M_l>threshold), with threshold defaulting to 1.

Important caveat
----------------
If the objective were only ``-log q(y|x)``, independent variational messages
could put arbitrary probability on the correct target without being BP-like.  To
avoid this degenerate solution, the default objective includes a BP-consistency
penalty:

    loss = CE(target, posterior)
           + bp_consistency_weight * mean_e KL(BP_update_e(m) || m_e)
           + message_l2_weight * mean_e ||c_e||^2.

The BP-update target is detached by default.  This makes the penalty a stable
successive-projection style objective.  Set ``detach_bp_targets=False`` if you
want gradients also through the BP update map; that is more literal but usually
less stable.

The resulting trajectory should be interpreted as an oracle message-space
training dynamics, not as a train/test generalization experiment of a neural net.
By default the messages are optimized on the same selected tasks on which the
curves are measured, exactly because messages are task-local inference objects,
not shared neural-network weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

# Reuse the existing BP/RHM implementation when the file is placed next to it.
from BP.constrained_bp_last_next_torch import (
    EPS,
    _as_device,
    normalize,
    centered_log_probs,
    build_train_test_dataset,
    upward_candidate_message,
    downward_candidate_for_child,
    encode_observations,
    possible_top_states_for_partial_block,
    torch_rules_to_numpy,
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _dtype_from_string(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}.")



def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()



def _center_logits(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)



def _softmax_from_logits(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(_center_logits(x), dim=-1)



def _safe_kl_p_to_q(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Mean KL(p || q) over all leading dimensions."""
    p = normalize(p, dim=-1, eps=eps)
    q = normalize(q, dim=-1, eps=eps)
    return (p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum(dim=-1).mean()



def _finite_mask_for_xy(x: np.ndarray, *ys: np.ndarray) -> np.ndarray:
    mask = np.isfinite(np.asarray(x, dtype=np.float64))
    for y in ys:
        yy = np.asarray(y)
        if yy.ndim == 1:
            mask &= np.isfinite(yy)
        else:
            mask &= np.all(np.isfinite(yy.reshape(yy.shape[0], -1)), axis=1)
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
    if not log:
        return
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if np.all(finite > 0):
        setter("log")
    else:
        setter("symlog", linthresh=_safe_linthresh(finite))



def _x_label(x_key: str) -> str:
    return {
        "step_values": "gradient step",
        "message_total_cost_mean": r"mean message cost $\langle\sum_e\|c_e\|^2\rangle$",
        "message_total_l2_norm_mean": r"mean summed message norm $\langle\sum_e\|c_e\|\rangle$",
        "message_global_l2_norm_mean": r"mean global message-logit norm $\langle(\sum_e\|c_e\|^2)^{1/2}\rangle$",
        "posterior_norm_mean": "posterior centered-logit norm",
    }.get(x_key, x_key)


# -----------------------------------------------------------------------------
# Prediction tasks: first, last, next
# -----------------------------------------------------------------------------


def make_crossentropy_prediction_tasks(
    sequences: torch.Tensor,
    *,
    mode: str,
    q: int,
    positions: Optional[Sequence[int]] = None,
    max_tasks: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Convert full RHM sequences [N,d] into masked prediction tasks.

    Supported modes:
      - first: mask token 0, observe all other leaves;
      - last:  mask token d-1, observe all previous leaves;
      - next:  for selected positions p, observe prefix <p and mask p:.

    The mask symbol is q.  Returned positions are zero-based.
    """
    if mode not in {"first", "last", "next"}:
        raise ValueError("mode must be 'first', 'last', or 'next'.")

    seq = sequences.to(torch.long)
    dev = seq.device
    N, d = int(seq.shape[0]), int(seq.shape[1])
    mask_symbol = int(q)

    if mode == "last":
        obs = seq.clone()
        obs[:, -1] = mask_symbol
        out = {
            "observations": obs,
            "target_pos": torch.full((N,), d - 1, dtype=torch.long, device=dev),
            "true_tokens": seq[:, -1].clone(),
            "seq_indices": torch.arange(N, dtype=torch.long, device=dev),
        }
    elif mode == "first":
        obs = seq.clone()
        obs[:, 0] = mask_symbol
        out = {
            "observations": obs,
            "target_pos": torch.zeros((N,), dtype=torch.long, device=dev),
            "true_tokens": seq[:, 0].clone(),
            "seq_indices": torch.arange(N, dtype=torch.long, device=dev),
        }
    else:
        if positions is None:
            pos_list = list(range(1, d))
        else:
            pos_list = [int(p) for p in positions]
            if any(p <= 0 or p >= d for p in pos_list):
                raise ValueError(f"next-token positions must lie in [1,{d-1}].")

        obs_list: List[torch.Tensor] = []
        target_pos: List[int] = []
        true_tokens: List[torch.Tensor] = []
        seq_indices: List[int] = []
        for p in pos_list:
            obs = seq.clone()
            obs[:, p:] = mask_symbol
            obs_list.append(obs)
            target_pos.extend([p] * N)
            true_tokens.append(seq[:, p].clone())
            seq_indices.extend(range(N))

        out = {
            "observations": torch.cat(obs_list, dim=0),
            "target_pos": torch.tensor(target_pos, dtype=torch.long, device=dev),
            "true_tokens": torch.cat(true_tokens, dim=0),
            "seq_indices": torch.tensor(seq_indices, dtype=torch.long, device=dev),
        }

    if max_tasks is not None and int(max_tasks) < int(out["observations"].shape[0]):
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        perm = torch.randperm(out["observations"].shape[0], device=dev, generator=g)[: int(max_tasks)]
        out = {k: v[perm] for k, v in out.items()}
    return out


# -----------------------------------------------------------------------------
# General hierarchy masks for masked first/last/next tasks
# -----------------------------------------------------------------------------


def candidate_set_from_observation_for_position_level(
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
    Candidate set A_{i,level} for any masked-observation pattern.

    Entries equal to q are treated as unobserved.  Inside the aligned level block
    containing target_pos, all observed leaves are clamped, the target is set to
    each candidate y, and future/other masked leaves remain unobserved.
    """
    if not (1 <= level <= L):
        raise ValueError(f"level must lie in [1,{L}], got {level}.")
    block_size = s**level
    block_start = (int(target_pos) // block_size) * block_size
    block_end = block_start + block_size
    block = np.full(block_size, -1, dtype=np.int64)

    for global_pos in range(block_start, block_end):
        val = int(observation[global_pos])
        if global_pos == int(target_pos):
            continue
        if 0 <= val < q:
            block[global_pos - block_start] = val

    rules_slice = rules_np[L - level : L]
    local_target = int(target_pos) - block_start
    out = np.zeros(q, dtype=bool)
    for y in range(q):
        b = block.copy()
        b[local_target] = y
        out[y] = possible_top_states_for_partial_block(b, rules_slice, s=s, q=q).size > 0
    return out



def precompute_hierarchy_masks_for_crossentropy_tasks(
    tasks: Dict[str, torch.Tensor],
    rules: Sequence[torch.Tensor],
    *,
    s: int,
    q: int,
    L: int,
) -> Dict[str, np.ndarray]:
    observations = tasks["observations"].detach().cpu().numpy().astype(np.int64, copy=False)
    target_pos = tasks["target_pos"].detach().cpu().numpy().astype(np.int64, copy=False)
    rules_np = torch_rules_to_numpy(rules)

    T = int(target_pos.shape[0])
    A_masks = np.zeros((T, L, q), dtype=bool)
    B_masks = np.zeros((T, L, q), dtype=bool)
    valid_masks = np.zeros((T, L), dtype=bool)
    all_vocab = np.ones(q, dtype=bool)

    for t in range(T):
        prev = all_vocab.copy()
        for ell in range(1, L + 1):
            A = candidate_set_from_observation_for_position_level(
                observations[t],
                int(target_pos[t]),
                ell,
                rules_np,
                s=s,
                q=q,
                L=L,
            )
            A = A & prev
            B = prev & (~A)
            A_masks[t, ell - 1] = A
            B_masks[t, ell - 1] = B
            valid_masks[t, ell - 1] = bool(A.any() and B.any())
            prev = A

    return {"A_masks": A_masks, "B_masks": B_masks, "valid_masks": valid_masks}


# -----------------------------------------------------------------------------
# Variational message model
# -----------------------------------------------------------------------------


class CrossEntropyBPMessageModel(nn.Module):
    """Trainable centered-logit BP messages for a fixed set of prediction tasks."""

    def __init__(
        self,
        observations: torch.Tensor,
        *,
        L: int,
        s: int,
        q: int,
        num_classes: int,
        init_logit_scale: float = 1e-3,
        seed_init: int = 0,
        dtype: torch.dtype = torch.float64,
        root_prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.observations = observations.to(torch.long)
        self.L = int(L)
        self.s = int(s)
        self.q = int(q)
        self.num_classes = int(num_classes)
        self.dtype = dtype
        self.device_ = observations.device

        if root_prior is None:
            rp = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype, device=self.device_)
        else:
            rp = normalize(root_prior.to(device=self.device_, dtype=dtype), dim=-1)
        self.register_buffer("root_prior", rp)

        T = int(observations.shape[0])
        g = torch.Generator(device=self.device_)
        g.manual_seed(int(seed_init))

        self.subtree_logits = nn.ParameterList()
        self.context_logits = nn.ParameterList()

        # Upward/subtree messages for depths 0,...,L-1.  Leaf evidence is fixed.
        for depth in range(self.L):
            K = self.num_classes if depth == 0 else self.q
            n_nodes = self.s**depth
            raw = torch.randn((T, n_nodes, K), generator=g, dtype=dtype, device=self.device_) * float(init_logit_scale)
            raw = _center_logits(raw)
            self.subtree_logits.append(nn.Parameter(raw))

        # Downward/context messages for depths 1,...,L.  Root context is fixed.
        for depth in range(1, self.L + 1):
            K = self.q
            n_nodes = self.s**depth
            raw = torch.randn((T, n_nodes, K), generator=g, dtype=dtype, device=self.device_) * float(init_logit_scale)
            raw = _center_logits(raw)
            self.context_logits.append(nn.Parameter(raw))

    def _select_obs(self, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if batch_idx is None:
            return self.observations
        return self.observations[batch_idx]

    def messages(self, batch_idx: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return context and subtree probability messages for the selected tasks."""
        obs = self._select_obs(batch_idx)
        B = int(obs.shape[0])

        subtree: List[torch.Tensor] = []
        for p in self.subtree_logits:
            x = p if batch_idx is None else p[batch_idx]
            subtree.append(_softmax_from_logits(x))
        leaves = encode_observations(obs, q=self.q, dtype=self.dtype)
        subtree.append(leaves)

        context: List[torch.Tensor] = []
        context.append(self.root_prior.reshape(1, 1, -1).expand(B, 1, self.num_classes))
        for p in self.context_logits:
            x = p if batch_idx is None else p[batch_idx]
            context.append(_softmax_from_logits(x))
        return context, subtree

    def posterior_at_targets(
        self,
        target_pos: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context, subtree = self.messages(batch_idx=batch_idx)
        obs = self._select_obs(batch_idx)
        pos = target_pos if batch_idx is None else target_pos[batch_idx]
        B = int(obs.shape[0])
        leaf_marginals = normalize(context[self.L] * subtree[self.L], dim=-1)
        return leaf_marginals[torch.arange(B, device=obs.device), pos]

    def message_norms_per_task(self, batch_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return per-task (sum ||c||^2, sum ||c||, sqrt(sum ||c||^2)).
        Norms are computed on centered raw logits, not on uncentered parameters.
        """
        T = self.observations.shape[0] if batch_idx is None else batch_idx.shape[0]
        total_cost = torch.zeros(T, dtype=self.dtype, device=self.device_)
        total_l2 = torch.zeros(T, dtype=self.dtype, device=self.device_)

        for plist in (self.subtree_logits, self.context_logits):
            for p in plist:
                x = p if batch_idx is None else p[batch_idx]
                c = _center_logits(x)
                n = torch.linalg.vector_norm(c, ord=2, dim=-1)  # [T,n_nodes]
                total_cost += (n**2).sum(dim=-1)
                total_l2 += n.sum(dim=-1)
        global_l2 = torch.sqrt(total_cost.clamp_min(0.0))
        return total_cost, total_l2, global_l2

    def bp_consistency_loss(
        self,
        rules: Sequence[torch.Tensor],
        rule_probs: Sequence[torch.Tensor],
        *,
        batch_idx: Optional[torch.Tensor] = None,
        detach_targets: bool = True,
        eps: float = 1e-30,
    ) -> torch.Tensor:
        """Average KL(BP_update(message) || message) over all internal messages."""
        context, subtree = self.messages(batch_idx=batch_idx)
        losses: List[torch.Tensor] = []

        # Upward/subtree consistency.
        for depth in range(self.L - 1, -1, -1):
            child_nodes = subtree[depth + 1]
            B = int(child_nodes.shape[0])
            n_nodes = self.s**depth
            child_block = child_nodes.reshape(B, n_nodes, self.s, -1)
            cand = upward_candidate_message(child_block, rules[depth], rule_probs[depth])
            if detach_targets:
                cand = cand.detach()
            losses.append(_safe_kl_p_to_q(cand, subtree[depth], eps=eps))

        # Downward/context consistency.  Root context is fixed and not penalized.
        for depth in range(0, self.L):
            B = int(context[depth].shape[0])
            n_nodes = self.s**depth
            children_subtree = subtree[depth + 1].reshape(B, n_nodes, self.s, -1)
            for t in range(self.s):
                cand = downward_candidate_for_child(
                    context[depth], children_subtree, rules[depth], rule_probs[depth], t
                )
                current = context[depth + 1].reshape(B, n_nodes, self.s, -1)[:, :, t, :]
                if detach_targets:
                    cand = cand.detach()
                losses.append(_safe_kl_p_to_q(cand, current, eps=eps))

        if len(losses) == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device_)
        return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _init_level_accumulators(L: int) -> Dict[str, np.ndarray]:
    return {
        "A_mass_sum": np.zeros(L, dtype=np.float64),
        "B_mass_sum": np.zeros(L, dtype=np.float64),
        "margin_sum": np.zeros(L, dtype=np.float64),
        "margin_pos_count": np.zeros(L, dtype=np.float64),
        "margin_gt_count": np.zeros(L, dtype=np.float64),
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
    *,
    margin_threshold: float = 1.0,
) -> Tuple[float, float]:
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
                acc["margin_gt_count"][ell] += float(margin > margin_threshold)
                acc["hier_acc_count"][ell] += float(A[int(pred[t])])
                acc["penalty_sum"][ell] += penalty
                acc["valid_count"][ell] += 1.0
                acc["penalty_all_sum"][ell] += penalty
                margins_for_recon.append(margin)
            else:
                acc["penalty_all_sum"][ell] += 0.0
                margins_for_recon.append(float("inf"))

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
def evaluate_crossentropy_bp_model(
    model: CrossEntropyBPMessageModel,
    tasks: Dict[str, torch.Tensor],
    hierarchy_masks: Dict[str, np.ndarray],
    *,
    eval_batch_size: int = 512,
    margin_threshold: float = 1.0,
) -> Dict[str, Any]:
    T = int(tasks["observations"].shape[0])
    L = model.L
    target_pos = tasks["target_pos"]
    true_tokens = tasks["true_tokens"]

    losses: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    posterior_norms: List[np.ndarray] = []
    total_costs: List[np.ndarray] = []
    total_l2s: List[np.ndarray] = []
    global_l2s: List[np.ndarray] = []
    acc = _init_level_accumulators(L)
    residual_sum = 0.0
    recon_err_sum = 0.0

    for start in range(0, T, int(eval_batch_size)):
        end = min(start + int(eval_batch_size), T)
        idx = torch.arange(start, end, device=model.device_, dtype=torch.long)
        posterior = model.posterior_at_targets(target_pos, batch_idx=idx)
        true_b = true_tokens[idx]
        p_true = posterior[torch.arange(end - start, device=model.device_), true_b].clamp_min(EPS)
        loss_b = -torch.log(p_true)
        pred = posterior.argmax(dim=-1)
        err_b = (pred != true_b).to(model.dtype)
        post_norm_b = torch.linalg.vector_norm(centered_log_probs(posterior, eps=EPS), ord=2, dim=-1)
        cost_b, l2_b, glob_b = model.message_norms_per_task(batch_idx=idx)

        losses.append(_to_numpy(loss_b))
        errors.append(_to_numpy(err_b))
        posterior_norms.append(_to_numpy(post_norm_b))
        total_costs.append(_to_numpy(cost_b))
        total_l2s.append(_to_numpy(l2_b))
        global_l2s.append(_to_numpy(glob_b))

        res_sum, rec_sum = _update_hierarchy_accumulators(
            acc,
            posterior=_to_numpy(posterior),
            pred=_to_numpy(pred),
            true_tokens=_to_numpy(true_b),
            A_masks=hierarchy_masks["A_masks"][start:end],
            B_masks=hierarchy_masks["B_masks"][start:end],
            valid_masks=hierarchy_masks["valid_masks"][start:end],
            margin_threshold=margin_threshold,
        )
        residual_sum += res_sum
        recon_err_sum += rec_sum

    loss_np = np.concatenate(losses)
    err_np = np.concatenate(errors)
    post_norm_np = np.concatenate(posterior_norms)
    cost_np = np.concatenate(total_costs)
    l2_np = np.concatenate(total_l2s)
    glob_np = np.concatenate(global_l2s)
    valid_count = acc["valid_count"]
    denom_valid = np.maximum(valid_count, 1.0)

    return {
        "loss_mean": float(loss_np.mean()),
        "loss_std": float(loss_np.std()),
        "error_mean": float(err_np.mean()),
        "error_std": float(err_np.std()),
        "posterior_norm_mean": float(post_norm_np.mean()),
        "posterior_norm_std": float(post_norm_np.std()),
        "message_total_cost_mean": float(cost_np.mean()),
        "message_total_cost_std": float(cost_np.std()),
        "message_total_l2_norm_mean": float(l2_np.mean()),
        "message_total_l2_norm_std": float(l2_np.std()),
        "message_global_l2_norm_mean": float(glob_np.mean()),
        "message_global_l2_norm_std": float(glob_np.std()),
        "A_mass_mean": acc["A_mass_sum"] / denom_valid,
        "B_mass_mean": acc["B_mass_sum"] / denom_valid,
        "margin_mean": acc["margin_sum"] / denom_valid,
        "margin_pos_frac": acc["margin_pos_count"] / denom_valid,
        "margin_gt1_frac": acc["margin_gt_count"] / denom_valid,
        "hier_acc": acc["hier_acc_count"] / denom_valid,
        "level_penalty_mean": acc["penalty_sum"] / denom_valid,
        "level_penalty_all_mean": acc["penalty_all_sum"] / float(T),
        "valid_level_frac": valid_count / float(T),
        "residual_mean": float(residual_sum / float(T)),
        "reconstructed_loss_abs_error_mean": float(recon_err_sum / float(T)),
    }


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------


def _make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError("optimizer must be one of: Adam, AdamW, SGD.")



def simulate_crossentropy_bp_trajectory(
    num_features: int = 32,
    num_classes: int = 32,
    num_synonyms: int = 8,
    tuple_size: int = 2,
    num_layers: int = 3,
    train_size: int = 32768,
    test_size: int = 2048,
    seed_rules: int = 0,
    seed_sample: int = 0,
    seed_init: int = 0,
    seed_batch: int = 0,
    prediction_mode: str = "last",
    positions: Optional[Sequence[int]] = None,
    max_test_sequences: Optional[int] = None,
    max_tasks: Optional[int] = None,
    task_seed: int = 0,
    zipf: Optional[float] = None,
    layer: Optional[int] = None,
    replacement: Optional[bool] = None,
    last_layer_powerlaw_a: Optional[float] = None,
    num_steps: int = 2000,
    eval_every: int = 20,
    lr: float = 1e-3,
    optimizer_name: str = "Adam",
    init_logit_scale: float = 1e-3,
    train_batch_size: Optional[int] = None,
    eval_batch_size: int = 512,
    bp_consistency_weight: float = 1.0,
    message_l2_weight: float = 0.0,
    detach_bp_targets: bool = True,
    grad_clip: Optional[float] = 10.0,
    margin_threshold: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run one cross-entropy BP-message gradient trajectory."""
    dev = _as_device(device)

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
    if int(test_sequences.shape[0]) == 0:
        raise ValueError("No test sequences are available.")

    tasks = make_crossentropy_prediction_tasks(
        test_sequences,
        mode=prediction_mode,
        q=num_features,
        positions=positions,
        max_tasks=max_tasks,
        seed=task_seed,
    )
    hierarchy_masks = precompute_hierarchy_masks_for_crossentropy_tasks(
        tasks,
        data["rules"],
        s=tuple_size,
        q=num_features,
        L=num_layers,
    )

    model = CrossEntropyBPMessageModel(
        tasks["observations"],
        L=num_layers,
        s=tuple_size,
        q=num_features,
        num_classes=num_classes,
        init_logit_scale=init_logit_scale,
        seed_init=seed_init,
        dtype=dtype,
    ).to(dev)

    opt = _make_optimizer(optimizer_name, model.parameters(), lr=lr)
    T = int(tasks["observations"].shape[0])
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed_batch))

    eval_steps: List[int] = []
    eval_records: List[Dict[str, Any]] = []
    objective_records: Dict[str, List[float]] = {
        "train_ce_loss": [],
        "train_consistency_loss": [],
        "train_message_l2_loss": [],
        "train_total_objective": [],
    }

    def evaluate_and_store(step: int) -> None:
        rec = evaluate_crossentropy_bp_model(
            model,
            tasks,
            hierarchy_masks,
            eval_batch_size=eval_batch_size,
            margin_threshold=margin_threshold,
        )
        eval_steps.append(int(step))
        eval_records.append(rec)

    evaluate_and_store(0)

    iterator = range(1, int(num_steps) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"crossentropy BP ({prediction_mode})")

    for step in iterator:
        if train_batch_size is None or int(train_batch_size) >= T:
            batch_idx = None
        else:
            batch_idx = torch.randint(0, T, size=(int(train_batch_size),), generator=g, device=dev)

        posterior = model.posterior_at_targets(tasks["target_pos"], batch_idx=batch_idx)
        true_tokens = tasks["true_tokens"] if batch_idx is None else tasks["true_tokens"][batch_idx]
        p_true = posterior[torch.arange(posterior.shape[0], device=dev), true_tokens].clamp_min(EPS)
        ce_loss = -torch.log(p_true).mean()
        consistency_loss = model.bp_consistency_loss(
            data["rules"],
            data["rule_probs"],
            batch_idx=batch_idx,
            detach_targets=detach_bp_targets,
        )
        msg_cost, _, _ = model.message_norms_per_task(batch_idx=batch_idx)
        l2_loss = msg_cost.mean()
        objective = ce_loss + float(bp_consistency_weight) * consistency_loss + float(message_l2_weight) * l2_loss

        opt.zero_grad(set_to_none=True)
        objective.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        opt.step()

        objective_records["train_ce_loss"].append(float(ce_loss.detach().cpu().item()))
        objective_records["train_consistency_loss"].append(float(consistency_loss.detach().cpu().item()))
        objective_records["train_message_l2_loss"].append(float(l2_loss.detach().cpu().item()))
        objective_records["train_total_objective"].append(float(objective.detach().cpu().item()))

        if step % int(eval_every) == 0 or step == int(num_steps):
            evaluate_and_store(step)

    def stack(key: str) -> np.ndarray:
        vals = [r[key] for r in eval_records]
        if isinstance(vals[0], np.ndarray):
            return np.stack(vals, axis=0)
        return np.asarray(vals, dtype=np.float64)

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
        "seed_init": seed_init,
        "seed_batch": seed_batch,
        "prediction_mode": prediction_mode,
        "positions": None if positions is None else list(map(int, positions)),
        "max_test_sequences": max_test_sequences,
        "max_tasks": max_tasks,
        "task_seed": task_seed,
        "zipf": zipf,
        "layer": layer,
        "replacement": replacement,
        "last_layer_powerlaw_a": last_layer_powerlaw_a,
        "num_steps": num_steps,
        "eval_every": eval_every,
        "lr": lr,
        "optimizer_name": optimizer_name,
        "init_logit_scale": init_logit_scale,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "bp_consistency_weight": bp_consistency_weight,
        "message_l2_weight": message_l2_weight,
        "detach_bp_targets": detach_bp_targets,
        "grad_clip": grad_clip,
        "margin_threshold": margin_threshold,
        "device": str(dev),
        "dtype": "float64" if dtype == torch.float64 else "float32",
    }

    result: Dict[str, Any] = {
        "step_values": np.asarray(eval_steps, dtype=np.int64),
        "loss_mean": stack("loss_mean"),
        "loss_std": stack("loss_std"),
        "error_mean": stack("error_mean"),
        "error_std": stack("error_std"),
        "posterior_norm_mean": stack("posterior_norm_mean"),
        "posterior_norm_std": stack("posterior_norm_std"),
        "message_total_cost_mean": stack("message_total_cost_mean"),
        "message_total_cost_std": stack("message_total_cost_std"),
        "message_total_l2_norm_mean": stack("message_total_l2_norm_mean"),
        "message_total_l2_norm_std": stack("message_total_l2_norm_std"),
        "message_global_l2_norm_mean": stack("message_global_l2_norm_mean"),
        "message_global_l2_norm_std": stack("message_global_l2_norm_std"),
        "A_mass_mean": stack("A_mass_mean"),
        "B_mass_mean": stack("B_mass_mean"),
        "margin_mean": stack("margin_mean"),
        "margin_pos_frac": stack("margin_pos_frac"),
        "margin_gt1_frac": stack("margin_gt1_frac"),
        "hier_acc": stack("hier_acc"),
        "level_penalty_mean": stack("level_penalty_mean"),
        "level_penalty_all_mean": stack("level_penalty_all_mean"),
        "valid_level_frac": stack("valid_level_frac"),
        "residual_mean": stack("residual_mean"),
        "reconstructed_loss_abs_error_mean": stack("reconstructed_loss_abs_error_mean"),
        "train_ce_loss": np.asarray(objective_records["train_ce_loss"], dtype=np.float64),
        "train_consistency_loss": np.asarray(objective_records["train_consistency_loss"], dtype=np.float64),
        "train_message_l2_loss": np.asarray(objective_records["train_message_l2_loss"], dtype=np.float64),
        "train_total_objective": np.asarray(objective_records["train_total_objective"], dtype=np.float64),
        "test_sequences": _to_numpy(test_sequences).astype(np.int64, copy=False),
        "task_target_pos": _to_numpy(tasks["target_pos"]).astype(np.int64, copy=False),
        "task_true_tokens": _to_numpy(tasks["true_tokens"]).astype(np.int64, copy=False),
        "task_seq_indices": _to_numpy(tasks["seq_indices"]).astype(np.int64, copy=False),
        "A_masks": hierarchy_masks["A_masks"],
        "B_masks": hierarchy_masks["B_masks"],
        "valid_masks": hierarchy_masks["valid_masks"],
        "params": params,
        "note": (
            "Cross-entropy gradient trajectory in BP-message space. Internal context/subtree messages are "
            "trainable centered logits initialized with small non-zero norm. x-axis should usually be "
            "message_global_l2_norm_mean or message_total_cost_mean, not step_values. The default objective "
            "uses CE plus KL(BP_update(m)||m) consistency; messages are task-local, so these curves are an "
            "oracle inference/training diagnostic rather than neural-network generalization curves. "
            "margin_gt1_frac stores Pr(M_l > margin_threshold), default threshold 1."
        ),
    }
    return result


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------


def save_results_npz(results: Dict[str, Any], out_prefix: str | Path) -> Path:
    out_prefix = Path(out_prefix)
    path = Path(str(out_prefix) + ".npz")
    np.savez_compressed(
        path,
        step_values=results["step_values"],
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
        message_global_l2_norm_mean=results["message_global_l2_norm_mean"],
        message_global_l2_norm_std=results["message_global_l2_norm_std"],
        A_mass_mean=results["A_mass_mean"],
        B_mass_mean=results["B_mass_mean"],
        margin_mean=results["margin_mean"],
        margin_pos_frac=results["margin_pos_frac"],
        margin_gt1_frac=results["margin_gt1_frac"],
        hier_acc=results["hier_acc"],
        level_penalty_mean=results["level_penalty_mean"],
        level_penalty_all_mean=results["level_penalty_all_mean"],
        valid_level_frac=results["valid_level_frac"],
        residual_mean=results["residual_mean"],
        reconstructed_loss_abs_error_mean=results["reconstructed_loss_abs_error_mean"],
        train_ce_loss=results["train_ce_loss"],
        train_consistency_loss=results["train_consistency_loss"],
        train_message_l2_loss=results["train_message_l2_loss"],
        train_total_objective=results["train_total_objective"],
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



def load_results(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
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



def plot_loss_error(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "message_global_l2_norm_mean",
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    results = _as_results(results_or_path)
    x = np.asarray(results[x_key], dtype=np.float64)
    loss = np.asarray(results["loss_mean"], dtype=np.float64)
    err = np.asarray(results["error_mean"], dtype=np.float64)
    mask = _finite_mask_for_xy(x, loss, err)
    x, loss, err = x[mask], loss[mask], err[mask]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(x, loss, marker="o")
    _set_axis_scale(axs[0], x, axis="x", log=log_x)
    _set_axis_scale(axs[0], loss, axis="y", log=log_y)
    axs[0].set_xlabel(_x_label(x_key))
    axs[0].set_ylabel("cross-entropy")
    axs[0].set_title("Loss")
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].plot(x, err, marker="o")
    _set_axis_scale(axs[1], x, axis="x", log=log_x)
    _set_axis_scale(axs[1], err, axis="y", log=log_y)
    axs[1].set_xlabel(_x_label(x_key))
    axs[1].set_ylabel("top-1 error")
    axs[1].set_title("Error")
    axs[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs



def plot_peeled_loss_and_margin_fraction(
    results_or_path: Dict[str, Any] | str | Path,
    x_key: str = "message_global_l2_norm_mean",
    loss_key: str = "level_penalty_mean",
    frac_key: str = "margin_gt1_frac",
    levels: Optional[Sequence[int]] = None,
    save_path: Optional[str | Path] = None,
    *,
    log_x: bool = True,
    log_y_left: bool = False,
    log_y_right: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side plot of peeled loss and margin-resolved fraction.

    By default the right plot is Pr(M_l>1), because frac_key='margin_gt1_frac'.
    Use frac_key='margin_pos_frac' for the old Pr(M_l>0) diagnostic.
    """
    results = _as_results(results_or_path)
    x = np.asarray(results[x_key], dtype=np.float64)
    peeled = np.asarray(results[loss_key], dtype=np.float64)
    frac = np.asarray(results[frac_key], dtype=np.float64)
    mask = _finite_mask_for_xy(x, peeled, frac)
    x, peeled, frac = x[mask], peeled[mask], frac[mask]

    n_levels = peeled.shape[1]
    if levels is None:
        levels = list(range(n_levels))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ell in levels:
        axs[0].plot(x, peeled[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _set_axis_scale(axs[0], x, axis="x", log=log_x)
    _set_axis_scale(axs[0], peeled[:, list(levels)], axis="y", log=log_y_left)
    axs[0].set_xlabel(_x_label(x_key))
    axs[0].set_ylabel(r"$\langle \log(1+e^{-M_\ell}) \rangle$")
    axs[0].set_title("Peeled loss")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(frameon=False)

    for ell in levels:
        axs[1].plot(x, frac[:, ell], marker="o", label=fr"$\ell={ell+1}$")
    _set_axis_scale(axs[1], x, axis="x", log=log_x)
    _set_axis_scale(axs[1], frac[:, list(levels)], axis="y", log=log_y_right)
    axs[1].set_xlabel(_x_label(x_key))
    ylabel = r"$\Pr(M_\ell>1)$" if frac_key == "margin_gt1_frac" else r"$\Pr(M_\ell>0)$"
    axs[1].set_ylabel(ylabel)
    axs[1].set_title("Resolved fraction")
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend(frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, axs



def plot_objective_terms(
    results_or_path: Dict[str, Any] | str | Path,
    save_path: Optional[str | Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    results = _as_results(results_or_path)
    steps = np.arange(1, len(results["train_total_objective"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["train_ce_loss", "train_consistency_loss", "train_message_l2_loss", "train_total_objective"]:
        if key in results:
            ax.plot(steps, np.asarray(results[key], dtype=np.float64), label=key)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("objective term")
    ax.set_title("Training objective diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_positions(s: Optional[str]) -> Optional[List[int]]:
    if s is None or not str(s).strip():
        return None
    return [int(x) for x in str(s).split(",") if x.strip()]



def _run_one_mode(args: argparse.Namespace, mode: str, out_prefix: str) -> Path:
    results = simulate_crossentropy_bp_trajectory(
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_synonyms=args.num_synonyms,
        tuple_size=args.tuple_size,
        num_layers=args.num_layers,
        train_size=args.train_size,
        test_size=args.test_size,
        seed_rules=args.seed_rules,
        seed_sample=args.seed_sample,
        seed_init=args.seed_init,
        seed_batch=args.seed_batch,
        prediction_mode=mode,
        positions=_parse_positions(args.positions),
        max_test_sequences=args.max_test_sequences,
        max_tasks=args.max_tasks,
        task_seed=args.task_seed,
        zipf=args.zipf,
        layer=args.layer,
        replacement=True if args.replacement else None,
        last_layer_powerlaw_a=args.last_layer_powerlaw_a,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        lr=args.lr,
        optimizer_name=args.optimizer,
        init_logit_scale=args.init_logit_scale,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        bp_consistency_weight=args.bp_consistency_weight,
        message_l2_weight=args.message_l2_weight,
        detach_bp_targets=not args.no_detach_bp_targets,
        grad_clip=args.grad_clip,
        margin_threshold=args.margin_threshold,
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
        show_progress=not args.no_progress,
    )
    path = save_results_npz(results, out_prefix)
    print("Saved", path)
    print(results["note"])
    print("params:", json.dumps(results["params"], indent=2, sort_keys=True))
    print("step_values:", results["step_values"])
    print("message_global_l2_norm_mean:", results["message_global_l2_norm_mean"])
    print("loss_mean:", results["loss_mean"])
    print("error_mean:", results["error_mean"])
    print("level_penalty_mean:\n", results["level_penalty_mean"])
    print("margin_gt1_frac:\n", results["margin_gt1_frac"])

    if args.make_plots:
        prefix = Path(out_prefix)
        plot_loss_error(results, x_key=args.x_key, save_path=str(prefix) + "_loss_error.png")
        plot_peeled_loss_and_margin_fraction(results, x_key=args.x_key, save_path=str(prefix) + "_peeled_margin.png")
        plot_objective_terms(results, save_path=str(prefix) + "_objective_terms.png")
        print("Saved plots with prefix", prefix)
    return path



def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-entropy gradient descent in RHM BP-message space.")
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--num_synonyms", type=int, default=8)
    parser.add_argument("--tuple_size", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=32768)
    parser.add_argument("--test_size", type=int, default=2048)
    parser.add_argument("--seed_rules", type=int, default=0)
    parser.add_argument("--seed_sample", type=int, default=0)
    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_batch", type=int, default=0)
    parser.add_argument("--prediction_mode", type=str, default="last", choices=["first", "last", "next", "both"])
    parser.add_argument("--positions", type=str, default=None, help="Comma-separated zero-based positions for next mode.")
    parser.add_argument("--max_test_sequences", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--task_seed", type=int, default=0)
    parser.add_argument("--zipf", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"])
    parser.add_argument("--init_logit_scale", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--bp_consistency_weight", type=float, default=1.0)
    parser.add_argument("--message_l2_weight", type=float, default=0.0)
    parser.add_argument("--no_detach_bp_targets", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--margin_threshold", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/crossentropy_BP")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--x_key", type=str, default="message_global_l2_norm_mean")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    if args.prediction_mode == "both":
        _run_one_mode(args, "first", args.out_prefix + "_first")
        _run_one_mode(args, "last", args.out_prefix + "_last")
    else:
        _run_one_mode(args, args.prediction_mode, args.out_prefix)


if __name__ == "__main__":
    main()
