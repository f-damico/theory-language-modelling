#!/usr/bin/env python3
"""
Variational / non-naive norm-constrained BP on the RHM for last-token and
next-token prediction.

This file keeps the same high-level interface and output dictionary as
`constrained_bp_last_next_torch.py`, but replaces the cheap centered-message
shrinkage rule

    c -> c / (1 + tau)

with an actual optimization over centered BP message logits.

Mathematical target
-------------------
For every prediction task t=(mu,i), exact unconstrained BP gives

    p_star_t(a) = P_RHM(X_i=a | prefix evidence).

The optimized messages c define an approximate target-leaf posterior q_t(a;c).
For a target average message budget lambda, this file approximately solves

    min_c  mean_t KL(p_star_t || q_t(. ; c)) + (gamma/2) R_BP(c)
    s.t.   mean_t sum_e ||c_{t,e}||_2^2 <= lambda,

where R_BP(c) is a soft BP-consistency penalty.  The constraint is enforced by
projecting the centered message logits onto the L2 budget ball after every
optimizer step.  This is still an approximation to the exact constrained
posterior problem, but it is not the one-shot shrinkage heuristic: messages are
optimized directly to match exact BP conditionals under a norm constraint.

Compatibility
-------------
The main notebook function has the same arguments as the previous script, plus
optimization controls:

    simulate_variational_constrained_bp_sweep(...)

The output keys are the same, so the existing plotting script can be reused.
A convenience alias `simulate_constrained_bp_sweep` is also provided.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

import BP.constrained_bp_last_next_torch as base

EPS = base.EPS


# -----------------------------------------------------------------------------
# Message-parameter utilities
# -----------------------------------------------------------------------------


def _center_param(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _params_from_bp_result(
    bp: base.BPBatchResult,
    *,
    L: int,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Initialize variational centered-log-message parameters from a BP result.

    Upward/subtree parameters are created for depths 0,...,L-1.
    Downward/context parameters are created for depths 1,...,L.
    Leaf evidence and root prior are fixed, not optimized.
    """
    sub_params: List[torch.nn.Parameter] = []
    ctx_params: List[torch.nn.Parameter] = []
    for depth in range(L):
        c = base.centered_log_probs(bp.subtree_messages[depth].detach())
        sub_params.append(torch.nn.Parameter(_center_param(c.clone())))
    for depth in range(1, L + 1):
        c = base.centered_log_probs(bp.context_messages[depth].detach())
        ctx_params.append(torch.nn.Parameter(_center_param(c.clone())))
    return sub_params, ctx_params


def _zero_params_like_bp_result(
    bp: base.BPBatchResult,
    *,
    L: int,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    sub_params: List[torch.nn.Parameter] = []
    ctx_params: List[torch.nn.Parameter] = []
    for depth in range(L):
        sub_params.append(torch.nn.Parameter(torch.zeros_like(bp.subtree_messages[depth])))
    for depth in range(1, L + 1):
        ctx_params.append(torch.nn.Parameter(torch.zeros_like(bp.context_messages[depth])))
    return sub_params, ctx_params


def _params_to_messages(
    sub_params: Sequence[torch.Tensor],
    ctx_params: Sequence[torch.Tensor],
    *,
    leaf_evidence: torch.Tensor,
    root_prior: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return subtree/context messages from centered-logit parameters."""
    L = len(sub_params)
    subtree: List[torch.Tensor] = []
    context: List[torch.Tensor] = []

    for p in sub_params:
        subtree.append(torch.softmax(_center_param(p), dim=-1))
    subtree.append(leaf_evidence)

    context.append(root_prior)
    for p in ctx_params:
        context.append(torch.softmax(_center_param(p), dim=-1))

    return subtree, context


def _message_cost_per_example(
    sub_params: Sequence[torch.Tensor],
    ctx_params: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Sum_e ||centered c_e||^2 for each example in the batch."""
    B = sub_params[0].shape[0]
    cost = torch.zeros(B, dtype=sub_params[0].dtype, device=sub_params[0].device)
    for p in list(sub_params) + list(ctx_params):
        c = _center_param(p)
        cost = cost + (c * c).sum(dim=tuple(range(1, c.ndim)))
    return cost


def _message_l2_sum_per_example(
    sub_params: Sequence[torch.Tensor],
    ctx_params: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Sum_e ||centered c_e||_2 for each example in the batch."""
    B = sub_params[0].shape[0]
    out = torch.zeros(B, dtype=sub_params[0].dtype, device=sub_params[0].device)
    for p in list(sub_params) + list(ctx_params):
        c = _center_param(p)
        per_message = torch.linalg.vector_norm(c, ord=2, dim=-1)
        out = out + per_message.sum(dim=tuple(range(1, per_message.ndim)))
    return out


@torch.no_grad()
def _project_params_to_budget(
    sub_params: Sequence[torch.nn.Parameter],
    ctx_params: Sequence[torch.nn.Parameter],
    *,
    lambda_total: float,
    budget_scope: str,
) -> None:
    """
    Project centered message logits onto the hard L2 budget.

    budget_scope='shared':       sum over the whole batch <= lambda_total * B.
    budget_scope='per_inference': each example <= lambda_total.
    """
    for p in list(sub_params) + list(ctx_params):
        p.data.copy_(_center_param(p.data))

    if not math.isfinite(float(lambda_total)):
        return

    if float(lambda_total) <= 0.0:
        for p in list(sub_params) + list(ctx_params):
            p.data.zero_()
        return

    cost = _message_cost_per_example(sub_params, ctx_params)
    if budget_scope == "shared":
        total_cost = cost.sum()
        cap = torch.tensor(float(lambda_total) * cost.numel(), dtype=cost.dtype, device=cost.device)
        if bool(total_cost > cap):
            scale = torch.sqrt(cap / total_cost.clamp_min(1e-30))
            for p in list(sub_params) + list(ctx_params):
                p.data.mul_(scale)
    elif budget_scope == "per_inference":
        cap = torch.tensor(float(lambda_total), dtype=cost.dtype, device=cost.device)
        scale = torch.ones_like(cost)
        bad = cost > cap
        scale[bad] = torch.sqrt(cap / cost[bad].clamp_min(1e-30))
        for p in list(sub_params) + list(ctx_params):
            view = scale
            while view.ndim < p.ndim:
                view = view.unsqueeze(-1)
            p.data.mul_(view)
    else:
        raise ValueError("budget_scope must be 'shared' or 'per_inference'.")

    for p in list(sub_params) + list(ctx_params):
        p.data.copy_(_center_param(p.data))


# -----------------------------------------------------------------------------
# Variational objective
# -----------------------------------------------------------------------------


def _root_prior_batch(
    *,
    B: int,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    root_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if root_prior is None:
        root = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype, device=device)
    else:
        root = base.normalize(root_prior.to(device=device, dtype=dtype), dim=-1)
    return root.reshape(1, 1, -1).expand(B, 1, num_classes).clone()


def _bp_consistency_penalty(
    sub_params: Sequence[torch.Tensor],
    ctx_params: Sequence[torch.Tensor],
    *,
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    leaf_evidence: torch.Tensor,
    root_prior_batched: torch.Tensor,
    s: int,
) -> torch.Tensor:
    """
    Soft penalty forcing variational messages to be near BP fixed-point updates.

    This keeps the optimization non-exponential while making the messages BP-like.
    """
    subtree, context = _params_to_messages(
        sub_params,
        ctx_params,
        leaf_evidence=leaf_evidence,
        root_prior=root_prior_batched,
    )
    L = len(rules)
    B = leaf_evidence.shape[0]
    penalty_sum = torch.zeros((), dtype=leaf_evidence.dtype, device=leaf_evidence.device)
    count = 0

    # Upward consistency: variational subtree message should match update from children.
    for depth in range(L - 1, -1, -1):
        n_nodes = s**depth
        child_block = subtree[depth + 1].reshape(B, n_nodes, s, -1)
        cand = base.upward_candidate_message(child_block, rules[depth], rule_probs[depth])
        cand_c = base.centered_log_probs(cand)
        diff = _center_param(sub_params[depth]) - cand_c
        penalty_sum = penalty_sum + (diff * diff).sum()
        count += diff.numel()

    # Downward consistency: variational context messages should match update from parent/siblings.
    for depth in range(0, L):
        n_nodes = s**depth
        children_subtree = subtree[depth + 1].reshape(B, n_nodes, s, -1)
        ctx_child = _center_param(ctx_params[depth]).reshape(B, n_nodes, s, -1)  # depth+1
        for t in range(s):
            cand = base.downward_candidate_for_child(
                context[depth], children_subtree, rules[depth], rule_probs[depth], t
            )
            cand_c = base.centered_log_probs(cand)
            diff = ctx_child[:, :, t, :] - cand_c
            penalty_sum = penalty_sum + (diff * diff).sum()
            count += diff.numel()

    return penalty_sum / max(count, 1)


def _posterior_from_params(
    sub_params: Sequence[torch.Tensor],
    ctx_params: Sequence[torch.Tensor],
    *,
    leaf_evidence: torch.Tensor,
    root_prior_batched: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    subtree, context = _params_to_messages(
        sub_params,
        ctx_params,
        leaf_evidence=leaf_evidence,
        root_prior=root_prior_batched,
    )
    prod = subtree[-1] * context[-1]
    leaf_marg = base.normalize(prod, dim=-1)
    batch_idx = torch.arange(leaf_evidence.shape[0], dtype=torch.long, device=leaf_evidence.device)
    return leaf_marg[batch_idx, target_pos, :]


def _forward_kl(p_star: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
    p = p_star.clamp_min(EPS)
    q = q_pred.clamp_min(EPS)
    return (p_star * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()


# -----------------------------------------------------------------------------
# Batch optimizer
# -----------------------------------------------------------------------------


def optimize_variational_bp_batch(
    rules: Sequence[torch.Tensor],
    rule_probs: Sequence[torch.Tensor],
    observations: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    lambda_total: float,
    budget_scope: str,
    q: int,
    s: int,
    num_classes: int,
    root_prior: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    num_opt_steps: int = 250,
    opt_lr: float = 5e-2,
    bp_consistency_weight: float = 10.0,
    optimizer_name: str = "adam",
    init_mode: str = "projected_exact",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Optimize message logits for one batch and one lambda.

    Returns posterior, exact p_star, optimized messages diagnostics, and loss history.
    """
    dev = observations.device
    B = observations.shape[0]
    L = len(rules)

    # Exact BP target p_star and initial BP messages.
    with torch.no_grad():
        exact = base.bp_pass_torch(
            rules,
            rule_probs,
            observations,
            q=q,
            s=s,
            num_classes=num_classes,
            tau=0.0,
            root_prior=root_prior,
            dtype=dtype,
        )
        leaf_exact = exact.marginals[-1]
        batch_idx = torch.arange(B, dtype=torch.long, device=dev)
        p_star = leaf_exact[batch_idx, target_pos, :].detach()

    if init_mode == "zero":
        sub_params, ctx_params = _zero_params_like_bp_result(exact, L=L)
    elif init_mode in {"exact", "projected_exact"}:
        sub_params, ctx_params = _params_from_bp_result(exact, L=L)
    else:
        raise ValueError("init_mode must be 'zero', 'exact', or 'projected_exact'.")

    if init_mode != "exact" or math.isfinite(float(lambda_total)):
        _project_params_to_budget(sub_params, ctx_params, lambda_total=lambda_total, budget_scope=budget_scope)

    leaf_evidence = base.encode_observations(observations, q=q, dtype=dtype)
    root_prior_batched = _root_prior_batch(
        B=B,
        num_classes=num_classes,
        device=dev,
        dtype=dtype,
        root_prior=root_prior,
    )

    params: List[torch.nn.Parameter] = list(sub_params) + list(ctx_params)
    name = optimizer_name.lower()
    if name == "adam":
        opt = torch.optim.Adam(params, lr=opt_lr)
    elif name == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=opt_lr, max_iter=20, line_search_fn="strong_wolfe")
    elif name == "sgd":
        opt = torch.optim.SGD(params, lr=opt_lr, momentum=0.9)
    else:
        raise ValueError("optimizer_name must be 'adam', 'sgd', or 'lbfgs'.")

    history: Dict[str, List[float]] = {"objective": [], "kl": [], "bp_penalty": [], "cost_mean": []}

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        q_pred = _posterior_from_params(
            sub_params,
            ctx_params,
            leaf_evidence=leaf_evidence,
            root_prior_batched=root_prior_batched,
            target_pos=target_pos,
        )
        kl = _forward_kl(p_star, q_pred)
        bp_pen = _bp_consistency_penalty(
            sub_params,
            ctx_params,
            rules=rules,
            rule_probs=rule_probs,
            leaf_evidence=leaf_evidence,
            root_prior_batched=root_prior_batched,
            s=s,
        )
        objective = kl + 0.5 * float(bp_consistency_weight) * bp_pen
        objective.backward()
        return objective

    for step in range(int(num_opt_steps)):
        if name == "lbfgs":
            loss_t = opt.step(closure)
        else:
            loss_t = closure()
            opt.step()
        _project_params_to_budget(sub_params, ctx_params, lambda_total=lambda_total, budget_scope=budget_scope)

        if step == 0 or step == int(num_opt_steps) - 1 or (verbose and step % max(1, num_opt_steps // 10) == 0):
            with torch.no_grad():
                q_tmp = _posterior_from_params(
                    sub_params,
                    ctx_params,
                    leaf_evidence=leaf_evidence,
                    root_prior_batched=root_prior_batched,
                    target_pos=target_pos,
                )
                kl_tmp = _forward_kl(p_star, q_tmp)
                bp_tmp = _bp_consistency_penalty(
                    sub_params,
                    ctx_params,
                    rules=rules,
                    rule_probs=rule_probs,
                    leaf_evidence=leaf_evidence,
                    root_prior_batched=root_prior_batched,
                    s=s,
                )
                cost_tmp = _message_cost_per_example(sub_params, ctx_params).mean()
                obj_tmp = kl_tmp + 0.5 * float(bp_consistency_weight) * bp_tmp
                history["objective"].append(float(obj_tmp.detach().cpu()))
                history["kl"].append(float(kl_tmp.detach().cpu()))
                history["bp_penalty"].append(float(bp_tmp.detach().cpu()))
                history["cost_mean"].append(float(cost_tmp.detach().cpu()))
                if verbose:
                    print(
                        f"step={step:04d} obj={history['objective'][-1]:.6g} "
                        f"kl={history['kl'][-1]:.6g} bp={history['bp_penalty'][-1]:.6g} "
                        f"cost={history['cost_mean'][-1]:.6g}"
                    )

    with torch.no_grad():
        posterior = _posterior_from_params(
            sub_params,
            ctx_params,
            leaf_evidence=leaf_evidence,
            root_prior_batched=root_prior_batched,
            target_pos=target_pos,
        )
        total_cost = _message_cost_per_example(sub_params, ctx_params)
        total_l2 = _message_l2_sum_per_example(sub_params, ctx_params)
        kl_final = _forward_kl(p_star, posterior)
        bp_final = _bp_consistency_penalty(
            sub_params,
            ctx_params,
            rules=rules,
            rule_probs=rule_probs,
            leaf_evidence=leaf_evidence,
            root_prior_batched=root_prior_batched,
            s=s,
        )

    return {
        "posterior": posterior.detach(),
        "p_star": p_star.detach(),
        "total_cost": total_cost.detach(),
        "total_l2_norm": total_l2.detach(),
        "kl_final": float(kl_final.detach().cpu()),
        "bp_penalty_final": float(bp_final.detach().cpu()),
        "history": history,
        "num_penalized_messages": int(exact.stats.num_penalized_messages),
    }


# -----------------------------------------------------------------------------
# Evaluation and sweep with same output keys as the shrinkage implementation
# -----------------------------------------------------------------------------


def evaluate_tasks_for_lambda_variational(
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
    num_opt_steps: int = 250,
    opt_lr: float = 5e-2,
    bp_consistency_weight: float = 10.0,
    optimizer_name: str = "adam",
    init_mode: str = "projected_exact",
    verbose_opt: bool = False,
) -> Dict[str, Any]:
    obs = tasks["observations"]
    target_pos = tasks["target_pos"]
    true_tokens = tasks["true_tokens"]
    T = int(obs.shape[0])
    L = len(rules)

    losses: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    posterior_norms: List[np.ndarray] = []
    total_costs: List[np.ndarray] = []
    total_l2s: List[np.ndarray] = []
    kl_values: List[float] = []
    bp_pen_values: List[float] = []

    acc = base._init_level_accumulators(L)
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

        out = optimize_variational_bp_batch(
            rules,
            rule_probs,
            obs_b,
            target_b,
            lambda_total=lambda_total,
            budget_scope=budget_scope,
            q=q,
            s=s,
            num_classes=num_classes,
            root_prior=root_prior,
            dtype=dtype,
            num_opt_steps=num_opt_steps,
            opt_lr=opt_lr,
            bp_consistency_weight=bp_consistency_weight,
            optimizer_name=optimizer_name,
            init_mode=init_mode,
            verbose=verbose_opt,
        )

        posterior = out["posterior"]
        batch_idx = torch.arange(obs_b.shape[0], device=obs_b.device)
        pred = torch.argmax(posterior, dim=-1)
        p_true = posterior[batch_idx, true_b].clamp_min(EPS)
        loss = -torch.log(p_true)
        err = (pred != true_b).to(dtype)
        post_norm = base.centered_logit_l2_norm_torch(posterior)

        losses.append(loss.detach().cpu().numpy())
        errors.append(err.detach().cpu().numpy())
        posterior_norms.append(post_norm.detach().cpu().numpy())
        total_costs.append(out["total_cost"].detach().cpu().numpy())
        total_l2s.append(out["total_l2_norm"].detach().cpu().numpy())
        kl_values.append(float(out["kl_final"]))
        bp_pen_values.append(float(out["bp_penalty_final"]))

        posterior_np = posterior.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy().astype(np.int64, copy=False)
        true_np = true_b.detach().cpu().numpy().astype(np.int64, copy=False)
        rs, re = base._update_hierarchy_accumulators(
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

    valid_count = acc["valid_count"]
    denom_valid = np.maximum(valid_count, 1.0)

    cap = float(lambda_total)
    if math.isfinite(cap):
        if budget_scope == "shared":
            hit = abs(float(total_costs_np.mean()) - cap) / max(cap, 1e-12) <= 5e-3 if cap > 0 else float(total_costs_np.mean()) <= 1e-8
        else:
            hit = float(np.mean(total_costs_np <= cap * (1.0 + 5e-3)))
    else:
        hit = 1.0

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
        # No tau in this method; keep keys for plotting compatibility.
        "tau_mean": float("nan"),
        "tau_std": float("nan"),
        "tau_shared": float("nan"),
        "shared_measured_cost": float(total_costs_np.mean()),
        "budget_hit_fraction": float(hit),
        "num_tasks": int(T),
        "num_penalized_messages": int(out["num_penalized_messages"]),
        "kl_to_exact_bp_mean": float(np.mean(kl_values)),
        "bp_penalty_mean": float(np.mean(bp_pen_values)),
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


def _stack_sweep_array(sweep: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = [r[key] for r in sweep]
    if isinstance(vals[0], np.ndarray):
        return np.stack(vals, axis=0)
    return np.asarray(vals)


def simulate_variational_constrained_bp_sweep(
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
    budget_tol_rel: float = 1e-6,  # kept for API compatibility; projection is used here.
    max_bisect_iter: int = 100,    # kept for API compatibility; unused here.
    num_opt_steps: int = 250,
    opt_lr: float = 5e-2,
    bp_consistency_weight: float = 10.0,
    optimizer_name: str = "adam",
    init_mode: str = "projected_exact",
    verbose_opt: bool = False,
) -> Dict[str, Any]:
    """
    Main notebook-friendly function.

    Same high-level signature/output as the shrinkage implementation, but this
    function optimizes centered message logits by projected gradient.
    """
    dev = base._as_device(device)
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-2, 2, 25), [np.inf]))
    lambda_values = np.asarray(lambda_values, dtype=np.float64)

    data = base.build_train_test_dataset(
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

    tasks = base.make_prediction_tasks(
        test_sequences,
        mode=prediction_mode,
        q=num_features,
        positions=positions,
        max_tasks=max_tasks,
        seed=task_seed,
    )

    hierarchy_masks = base.precompute_hierarchy_masks_for_tasks(
        test_sequences,
        tasks,
        data["rules"],
        s=tuple_size,
        q=num_features,
        L=num_layers,
    )

    sweep: List[Dict[str, Any]] = []
    iterator = tqdm(lambda_values, desc=f"variational {prediction_mode} BP sweep ({budget_scope})")
    for lam in iterator:
        out = evaluate_tasks_for_lambda_variational(
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
            num_opt_steps=num_opt_steps,
            opt_lr=opt_lr,
            bp_consistency_weight=bp_consistency_weight,
            optimizer_name=optimizer_name,
            init_mode=init_mode,
            verbose_opt=verbose_opt,
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
            "num_opt_steps": int(num_opt_steps),
            "opt_lr": float(opt_lr),
            "bp_consistency_weight": float(bp_consistency_weight),
            "optimizer_name": str(optimizer_name),
            "init_mode": str(init_mode),
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
        "kl_to_exact_bp_mean": _stack_sweep_array(sweep, "kl_to_exact_bp_mean"),
        "bp_penalty_mean": _stack_sweep_array(sweep, "bp_penalty_mean"),
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
            "Variational norm-constrained BP sweep for last-token or next-token prediction. "
            "Unlike the shrinkage file, this optimizes centered message logits to reduce "
            "KL(p_star || q) to exact BP under a hard projected message-norm budget, plus a "
            "soft BP-consistency penalty. Existing plotting code can be reused."
        ),
    }
    return result


# Convenience alias if the user wants the old call name after changing import file.
simulate_constrained_bp_sweep = simulate_variational_constrained_bp_sweep


def save_results_npz(results: Dict[str, Any], out_prefix: str | Path) -> Path:
    """Save results with the same keys as the shrinkage implementation plus variational diagnostics."""
    out_prefix = Path(out_prefix)
    path = Path(str(out_prefix) + ".npz")
    payload = dict(
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
        kl_to_exact_bp_mean=results["kl_to_exact_bp_mean"],
        bp_penalty_mean=results["bp_penalty_mean"],
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
    np.savez_compressed(path, **payload)
    return path


def _parse_lambda_values_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.lambda_values is not None:
        return np.array([float(x) for x in args.lambda_values.split(",") if x.strip()], dtype=np.float64)
    vals = np.logspace(args.lambda_log10_min, args.lambda_log10_max, args.lambda_num)
    if args.include_zero:
        vals = np.concatenate(([0.0], vals))
    if args.include_inf:
        vals = np.concatenate((vals, [np.inf]))
    return vals.astype(np.float64)


def _dtype_from_string(name: str) -> torch.dtype:
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError("dtype must be float64 or float32.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Variational constrained BP sweep for RHM last/next-token prediction.")
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
    parser.add_argument("--positions", type=str, default=None)
    parser.add_argument("--max_test_sequences", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--task_seed", type=int, default=0)
    parser.add_argument("--zipf", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--last_layer_powerlaw_a", type=float, default=None)
    parser.add_argument("--lambda_values", type=str, default=None)
    parser.add_argument("--lambda_log10_min", type=float, default=-2.0)
    parser.add_argument("--lambda_log10_max", type=float, default=2.0)
    parser.add_argument("--lambda_num", type=int, default=25)
    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--include_inf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--num_opt_steps", type=int, default=250)
    parser.add_argument("--opt_lr", type=float, default=5e-2)
    parser.add_argument("--bp_consistency_weight", type=float, default=10.0)
    parser.add_argument("--optimizer_name", type=str, default="adam", choices=["adam", "sgd", "lbfgs"])
    parser.add_argument("--init_mode", type=str, default="projected_exact", choices=["zero", "exact", "projected_exact"])
    parser.add_argument("--verbose_opt", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/variational_constrained_bp_last_next_torch")
    args = parser.parse_args()

    lambda_values = _parse_lambda_values_from_args(args)
    positions = None
    if args.positions is not None:
        positions = [int(x) for x in args.positions.split(",") if x.strip()]

    results = simulate_variational_constrained_bp_sweep(
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
        num_opt_steps=args.num_opt_steps,
        opt_lr=args.opt_lr,
        bp_consistency_weight=args.bp_consistency_weight,
        optimizer_name=args.optimizer_name,
        init_mode=args.init_mode,
        verbose_opt=args.verbose_opt,
    )

    path = save_results_npz(results, args.out_prefix)
    print("Saved", path)
    print(results["note"])
    print("params:", json.dumps(results["params"], indent=2, sort_keys=True))
    print("lambda_values:", results["lambda_values"])
    print("message_total_cost_mean:", results["message_total_cost_mean"])
    print("loss_mean:", results["loss_mean"])
    print("error_mean:", results["error_mean"])
    print("kl_to_exact_bp_mean:", results["kl_to_exact_bp_mean"])
    print("bp_penalty_mean:", results["bp_penalty_mean"])
    print("margin_pos_frac:\n", results["margin_pos_frac"])
    print("level_penalty_mean:\n", results["level_penalty_mean"])


if __name__ == "__main__":
    main()
