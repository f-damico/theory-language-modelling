"""
Belief propagation on an RHM tree with a *global* Lagrangian budget.

Main idea
---------
The original constrained implementation enforced a *local* constraint by
projecting every internal message independently onto a centered-logit L2 ball.

Here we switch to the simpler global-Lagrangian alternative:

1) keep the usual BP upward/downward updates,
2) after each exact BP candidate message m is computed, write it as

       log m = const + c,

   where c is the centered log-message,
3) shrink *all* internal messages with the *same* dual variable tau,

       c -> c / (1 + tau),

   which is the natural proximal shrinkage for a quadratic penalty
   (tau / 2) * ||c||_2^2,
4) choose tau by bisection so that the *total* quadratic message cost

       sum_e ||c_e||_2^2

   matches, as closely as possible, a target global budget `lambda_total`.

Thus:
- lambda_total is the target *total* message cost budget per inference problem,
- tau is the corresponding shared dual variable,
- lambda_total = 0 forces all propagated internal messages to be uniform,
- lambda_total -> +inf recovers exact oracle BP (tau -> 0).

This is still not an explicit free-energy solver over beliefs. It is the
lightest modification of the existing BP code that turns the old local clip into
one shared global budget mechanism while preserving the same inference flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class MessageStats:
    """Diagnostics for one BP inference problem."""

    tau: float = 0.0
    total_cost: float = 0.0
    total_l2_norm: float = 0.0
    num_penalized_messages: int = 0
    converged_to_budget: bool = True

    def as_dict(self) -> Dict[str, float | int | bool]:
        return {
            "tau": float(self.tau),
            "total_cost": float(self.total_cost),
            "total_l2_norm": float(self.total_l2_norm),
            "num_penalized_messages": int(self.num_penalized_messages),
            "converged_to_budget": bool(self.converged_to_budget),
        }


def _node_state_dim(depth: int, l: int, q: int, num_classes: int) -> int:
    """Return the alphabet size at a given tree depth from the root."""
    return num_classes if depth == 0 else q


def _normalize(vec: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    total = float(np.sum(vec))
    if not np.isfinite(total) or total <= eps:
        vec = np.ones_like(vec, dtype=np.float64)
        total = float(np.sum(vec))
    return vec / total




def _normalize_rows(mat: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Row-wise normalization for 2D arrays, with uniform fallback."""
    mat = np.asarray(mat, dtype=np.float64)
    totals = np.sum(mat, axis=1, keepdims=True)
    bad = (~np.isfinite(totals)) | (totals <= eps)
    if np.any(bad):
        mat = mat.copy()
        mat[bad[:, 0]] = 1.0
        totals = np.sum(mat, axis=1, keepdims=True)
    return mat / totals

def _centered_log_message(msg: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Return centered log-message logits c = log m - mean(log m)."""
    msg = _normalize(np.asarray(msg, dtype=np.float64), eps=eps)
    logm = np.log(np.clip(msg, eps, 1.0))
    return logm - np.mean(logm)


def _softmax_from_centered_logits(centered: np.ndarray) -> np.ndarray:
    centered = np.asarray(centered, dtype=np.float64)
    centered = centered - np.max(centered)
    out = np.exp(centered)
    return _normalize(out)


def _apply_tau_shrinkage(
    msg: np.ndarray,
    tau: float,
    eps: float = 1e-300,
) -> Tuple[np.ndarray, float, float]:
    """
    Apply the shared quadratic-penalty shrinkage to one probability vector.

    For a candidate message m with centered logits c, the penalized message uses

        c_tau = c / (1 + tau).

    The returned diagnostics are computed on the *final* shrunk message:
    - cost = ||c_tau||_2^2,
    - l2_norm = ||c_tau||_2.
    """
    msg = _normalize(np.asarray(msg, dtype=np.float64), eps=eps)

    if not np.isfinite(tau):
        uniform = np.full_like(msg, 1.0 / msg.size, dtype=np.float64)
        return uniform, 0.0, 0.0

    if tau <= 0.0:
        centered = _centered_log_message(msg, eps=eps)
        norm = float(np.linalg.norm(centered))
        return msg, norm * norm, norm

    centered = _centered_log_message(msg, eps=eps)
    shrunk = centered / (1.0 + tau)
    out = _softmax_from_centered_logits(shrunk)
    norm = float(np.linalg.norm(shrunk))
    return out, norm * norm, norm




def _prepare_rule_probs(
    rules: Sequence[np.ndarray],
    rule_probs: Optional[Sequence[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Return one probability matrix per level with shape [num_parents, m].

    If `rule_probs` is None, all rules are taken equiprobable, reproducing the
    original repo behaviour exactly.
    """
    out: List[np.ndarray] = []
    if rule_probs is None:
        for level_rules in rules:
            num_parents, m = level_rules.shape[:2]
            out.append(np.full((num_parents, m), 1.0 / m, dtype=np.float64))
        return out

    if len(rule_probs) != len(rules):
        raise ValueError("rule_probs must have the same number of levels as rules.")

    for level, (level_rules, level_probs) in enumerate(zip(rules, rule_probs)):
        num_parents, m = level_rules.shape[:2]
        arr = np.asarray(level_probs, dtype=np.float64)
        if arr.shape != (num_parents, m):
            raise ValueError(
                f"rule_probs[{level}] has shape {arr.shape}, expected {(num_parents, m)}"
            )
        arr = _normalize_rows(arr)
        out.append(arr)
    return out

def generate_tree(
    l: int,
    q: int,
    leaves: np.ndarray,
    num_classes: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Initialize context messages (`up_messages`) and subtree/evidence messages
    (`down_messages`).

    Shapes differ by depth because the root can have size `num_classes` while all
    lower levels have size `q`.
    """
    num_classes = q if num_classes is None else num_classes
    up_messages: List[np.ndarray] = []
    down_messages: List[np.ndarray] = []

    s = int(round(leaves.shape[0] ** (1.0 / l))) if l > 0 else 1

    for depth in range(l + 1):
        state_dim = _node_state_dim(depth, l, q, num_classes)
        n_nodes = s**depth
        up_messages.append(np.full((n_nodes, state_dim), 1.0 / state_dim, dtype=np.float64))
        down_messages.append(np.full((n_nodes, state_dim), 1.0 / state_dim, dtype=np.float64))

    if leaves.shape != down_messages[-1].shape:
        raise ValueError(
            f"Leaves have shape {leaves.shape}, expected {down_messages[-1].shape}"
        )

    down_messages[-1] = leaves.astype(np.float64, copy=True)
    return up_messages, down_messages


def _run_bp_with_tau(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xis: Sequence[int],
    s: int,
    tau: float = 0.0,
    num_classes: Optional[int] = None,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
) -> Tuple[List[np.ndarray], MessageStats]:
    """
    Run one BP inference problem with a *fixed* shared dual variable tau.

    Returns node marginals plus message diagnostics.
    """
    num_classes = q if num_classes is None else num_classes
    root_dim = _node_state_dim(0, l, q, num_classes)

    if root_prior is None:
        root_prior_arr = np.full(root_dim, 1.0 / root_dim, dtype=np.float64)
    else:
        root_prior_arr = _normalize(np.asarray(root_prior, dtype=np.float64))

    prepared_rule_probs = _prepare_rule_probs(rules, rule_probs=rule_probs)

    leaves_BP = _encode_leaves(xis, q=q, mask_symbol=mask_symbol)
    up_messages, down_messages = generate_tree(l=l, q=q, leaves=leaves_BP, num_classes=num_classes)

    total_cost = 0.0
    total_l2_norm = 0.0
    num_penalized_messages = 0

    # Upward pass: leaves -> root.
    for depth in range(l - 1, -1, -1):
        parent_dim = _node_state_dim(depth, l, q, num_classes)
        child_nodes = down_messages[depth + 1]
        parent_nodes = down_messages[depth]
        rule_tensor = np.asarray(rules[depth], dtype=np.int64)
        rule_prob_tensor = prepared_rule_probs[depth]
        n_nodes = parent_nodes.shape[0]
        m = rule_tensor.shape[1]

        for j in range(n_nodes):
            children_block = child_nodes[j * s : (j + 1) * s]
            msg = np.zeros(parent_dim, dtype=np.float64)
            for parent_state in range(parent_dim):
                total = 0.0
                for r in range(m):
                    prod = float(rule_prob_tensor[parent_state, r])
                    for t in range(s):
                        child_state = rule_tensor[parent_state, r, t]
                        prod *= children_block[t, child_state]
                    total += prod
                msg[parent_state] = total
            msg = _normalize(msg)
            msg, cost, l2_norm = _apply_tau_shrinkage(msg, tau=tau)
            parent_nodes[j] = msg
            total_cost += cost
            total_l2_norm += l2_norm
            num_penalized_messages += 1

    # Root context message: fixed prior, not penalized.
    up_messages[0][0] = root_prior_arr

    # Downward pass: root -> leaves.
    for depth in range(0, l):
        parent_dim = _node_state_dim(depth, l, q, num_classes)
        child_dim = _node_state_dim(depth + 1, l, q, num_classes)
        rule_tensor = np.asarray(rules[depth], dtype=np.int64)
        rule_prob_tensor = prepared_rule_probs[depth]
        m = rule_tensor.shape[1]

        for j in range(up_messages[depth].shape[0]):
            parent_context = up_messages[depth][j]
            children_block_down = down_messages[depth + 1][j * s : (j + 1) * s]
            for child_pos in range(s):
                child_msg = np.zeros(child_dim, dtype=np.float64)
                for child_state in range(child_dim):
                    total = 0.0
                    for parent_state in range(parent_dim):
                        parent_weight = parent_context[parent_state]
                        if parent_weight <= 0.0:
                            continue
                        for r in range(m):
                            if rule_tensor[parent_state, r, child_pos] != child_state:
                                continue
                            weight = parent_weight * float(rule_prob_tensor[parent_state, r])
                            for sib_pos in range(s):
                                if sib_pos == child_pos:
                                    continue
                                sib_state = rule_tensor[parent_state, r, sib_pos]
                                weight *= children_block_down[sib_pos, sib_state]
                            total += weight
                    child_msg[child_state] = total
                child_msg = _normalize(child_msg)
                child_msg, cost, l2_norm = _apply_tau_shrinkage(child_msg, tau=tau)
                up_messages[depth + 1][j * s + child_pos] = child_msg
                total_cost += cost
                total_l2_norm += l2_norm
                num_penalized_messages += 1

    marginals = compute_marginals(l=l, up_messages=up_messages, down_messages=down_messages)
    stats = MessageStats(
        tau=float(tau),
        total_cost=float(total_cost),
        total_l2_norm=float(total_l2_norm),
        num_penalized_messages=int(num_penalized_messages),
        converged_to_budget=True,
    )
    return marginals, stats


def _solve_tau_for_budget(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xis: Sequence[int],
    s: int,
    lambda_total: float,
    num_classes: Optional[int] = None,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
    tau_guess: Optional[float] = None,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 60,
) -> Tuple[List[np.ndarray], MessageStats]:
    """
    Choose the shared dual variable tau so that the final total quadratic
    message cost approximately matches the target budget lambda_total.

    If the exact BP solution already uses total_cost <= lambda_total, tau = 0 is
    returned.
    """
    if not np.isfinite(lambda_total):
        return _run_bp_with_tau(
            rules=rules,
            l=l,
            q=q,
            xis=xis,
            s=s,
            tau=0.0,
            num_classes=num_classes,
            rule_probs=rule_probs,
            root_prior=root_prior,
            mask_symbol=mask_symbol,
        )

    if lambda_total <= 0.0:
        # Use a very large but finite tau so diagnostics stay finite.
        marginals, stats = _run_bp_with_tau(
            rules=rules,
            l=l,
            q=q,
            xis=xis,
            s=s,
            tau=1.0e12,
            num_classes=num_classes,
            rule_probs=rule_probs,
            root_prior=root_prior,
            mask_symbol=mask_symbol,
        )
        stats.total_cost = 0.0
        stats.total_l2_norm = 0.0
        stats.converged_to_budget = True
        return marginals, stats

    marginals0, stats0 = _run_bp_with_tau(
        rules=rules,
        l=l,
        q=q,
        xis=xis,
        s=s,
        tau=0.0,
        num_classes=num_classes,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
    )
    if stats0.total_cost <= lambda_total:
        stats0.converged_to_budget = True
        return marginals0, stats0

    tau_lo = 0.0
    tau_hi = 1.0
    marginals_hi, stats_hi = _run_bp_with_tau(
        rules=rules,
        l=l,
        q=q,
        xis=xis,
        s=s,
        tau=tau_hi,
        num_classes=num_classes,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
    )

    expand_iter = 0
    while stats_hi.total_cost > lambda_total and expand_iter < max_bisect_iter:
        tau_hi *= 2.0
        marginals_hi, stats_hi = _run_bp_with_tau(
            rules=rules,
            l=l,
            q=q,
            xis=xis,
            s=s,
            tau=tau_hi,
            num_classes=num_classes,
            rule_probs=rule_probs,
            root_prior=root_prior,
            mask_symbol=mask_symbol,
        )
        expand_iter += 1

    best_marginals = marginals_hi
    best_stats = stats_hi
    best_gap = abs(best_stats.total_cost - lambda_total)

    for _ in range(max_bisect_iter):
        tau_mid = 0.5 * (tau_lo + tau_hi)
        marginals_mid, stats_mid = _run_bp_with_tau(
            rules=rules,
            l=l,
            q=q,
            xis=xis,
            s=s,
            tau=tau_mid,
            num_classes=num_classes,
            rule_probs=rule_probs,
            root_prior=root_prior,
            mask_symbol=mask_symbol,
        )
        gap = abs(stats_mid.total_cost - lambda_total)
        if gap < best_gap:
            best_gap = gap
            best_marginals = marginals_mid
            best_stats = stats_mid

        rel_gap = gap / max(lambda_total, 1e-12)
        if rel_gap <= budget_tol_rel or (tau_hi - tau_lo) <= tau_tol:
            best_stats.converged_to_budget = True
            return best_marginals, best_stats

        if stats_mid.total_cost > lambda_total:
            tau_lo = tau_mid
        else:
            tau_hi = tau_mid

    best_stats.converged_to_budget = False
    return best_marginals, best_stats


def compute_marginals(
    l: int,
    up_messages: List[np.ndarray],
    down_messages: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute node marginals by multiplying incoming messages."""
    marginals: List[np.ndarray] = []
    for depth in range(l + 1):
        prod = up_messages[depth] * down_messages[depth]
        out = np.empty_like(prod)
        for j in range(prod.shape[0]):
            out[j] = _normalize(prod[j])
        marginals.append(out)
    return marginals


def _encode_leaves(
    xis: Sequence[int],
    q: int,
    mask_symbol: Optional[int] = None,
) -> np.ndarray:
    """Turn leaf observations into one-hot / uniform BP evidence vectors."""
    mask_symbol = q if mask_symbol is None else mask_symbol

    xis = np.asarray(xis, dtype=np.int64)
    leaves = np.empty((len(xis), q), dtype=np.float64)
    for i, x in enumerate(xis):
        if x == mask_symbol:
            leaves[i] = 1.0 / q
        else:
            leaves[i] = 0.0
            leaves[i, int(x)] = 1.0
    return leaves


def run_BP(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xis: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_total: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
    tau_guess: Optional[float] = None,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 60,
) -> Tuple[List[np.ndarray], MessageStats]:
    """
    Run BP with a global quadratic budget on all internal centered log-messages.

    `rule_probs[level][parent, r]` gives the probability of choosing rule `r`
    for a given parent state at that level. If omitted, all rules are equiprobable.

    Returns:
        marginals: list of arrays of node marginals by tree depth,
        stats: diagnostics including the chosen tau and the realized total cost.
    """
    return _solve_tau_for_budget(
        rules=rules,
        l=l,
        q=q,
        xis=xis,
        s=s,
        lambda_total=lambda_total,
        num_classes=num_classes,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
        tau_guess=tau_guess,
        tau_tol=tau_tol,
        budget_tol_rel=budget_tol_rel,
        max_bisect_iter=max_bisect_iter,
    )


def masked_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xis: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_total: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
    tau_guess: Optional[float] = None,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 60,
) -> Tuple[np.ndarray, MessageStats]:
    """Return the leaf marginals for a masked sequence plus message diagnostics."""
    marginals, stats = run_BP(
        rules=rules,
        l=l,
        q=q,
        xis=xis,
        s=s,
        num_classes=num_classes,
        lambda_total=lambda_total,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
        tau_guess=tau_guess,
        tau_tol=tau_tol,
        budget_tol_rel=budget_tol_rel,
        max_bisect_iter=max_bisect_iter,
    )
    return marginals[-1], stats


def last_token_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xi: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_total: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
    tau_guess: Optional[float] = None,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 60,
) -> Tuple[np.ndarray, MessageStats]:
    """Return the posterior of the last token after masking it."""
    xi_masked = np.asarray(xi, dtype=np.int64).copy()
    mask_symbol = q if mask_symbol is None else mask_symbol
    xi_masked[-1] = mask_symbol
    leaf_marginals, stats = masked_inference(
        rules=rules,
        l=l,
        q=q,
        xis=xi_masked,
        s=s,
        num_classes=num_classes,
        lambda_total=lambda_total,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
        tau_guess=tau_guess,
        tau_tol=tau_tol,
        budget_tol_rel=budget_tol_rel,
        max_bisect_iter=max_bisect_iter,
    )
    return leaf_marginals[-1], stats


def run_last_token_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xi: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_total: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
    tau_guess: Optional[float] = None,
    tau_tol: float = 1e-8,
    budget_tol_rel: float = 1e-4,
    max_bisect_iter: int = 60,
) -> Tuple[np.ndarray, int, float, Dict[str, float | int | bool]]:
    """
    Convenience wrapper for next-token prediction.

    Returns:
        posterior,
        predicted_token,
        loss_on_true_last_token,
        diagnostics dict with tau and realized message-cost statistics.
    """
    xi = np.asarray(xi, dtype=np.int64)
    posterior, stats = last_token_inference(
        rules=rules,
        l=l,
        q=q,
        xi=xi,
        s=s,
        num_classes=num_classes,
        lambda_total=lambda_total,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
        tau_guess=tau_guess,
        tau_tol=tau_tol,
        budget_tol_rel=budget_tol_rel,
        max_bisect_iter=max_bisect_iter,
    )
    pred = int(np.argmax(posterior))
    loss = float(-np.log(np.clip(posterior[int(xi[-1])], 1e-300, 1.0)))
    return posterior, pred, loss, stats.as_dict()
