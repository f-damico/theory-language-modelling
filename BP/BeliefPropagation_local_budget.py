"""
Functions to implement constrained belief propagation on an RHM tree with an
optional non-uniform rule distribution on any chosen layer.

This is the local-budget version: every internal message is projected
independently onto a centered-logit L2 ball of radius `lambda_msg`.

Compared to the original local-budget BP file, this version additionally
supports `rule_probs[level][parent, r]`, matching the repo's Zipf-style dataset
construction when a selected layer uses non-uniform rule probabilities.
If `rule_probs` is None, the original equal-probability behaviour is recovered
exactly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray


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


def _project_message_strength(
    msg: np.ndarray,
    lambda_msg: float,
    eps: float = 1e-300,
) -> np.ndarray:
    """
    Project a probability vector onto the centered-logit L2 ball of radius
    lambda_msg.

    Let c = log(m) - mean(log(m)). If ||c||_2 <= lambda_msg, return m.
    Otherwise, radially rescale c to norm lambda_msg and map back with softmax.
    """
    msg = _normalize(np.asarray(msg, dtype=np.float64), eps=eps)
    if not np.isfinite(lambda_msg):
        return msg
    if lambda_msg <= 0.0:
        return np.full_like(msg, 1.0 / msg.size, dtype=np.float64)

    logm = np.log(np.clip(msg, eps, 1.0))
    centered = logm - np.mean(logm)
    norm = float(np.linalg.norm(centered))
    if norm <= lambda_msg:
        return msg

    centered = centered * (lambda_msg / norm)
    centered -= np.max(centered)
    out = np.exp(centered)
    return _normalize(out, eps=eps)



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



def update_messages(
    l: int,
    q: int,
    up_messages: List[np.ndarray],
    down_messages: List[np.ndarray],
    rules: Sequence[np.ndarray],
    s: int,
    lambda_msg: float = np.inf,
    num_classes: Optional[int] = None,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Update messages on the RHM tree with optional non-uniform rule probabilities.
    """
    num_classes = q if num_classes is None else num_classes
    root_dim = _node_state_dim(0, l, q, num_classes)

    if root_prior is None:
        root_prior = np.full(root_dim, 1.0 / root_dim, dtype=np.float64)
    else:
        root_prior = _normalize(np.asarray(root_prior, dtype=np.float64))

    prepared_rule_probs = _prepare_rule_probs(rules, rule_probs=rule_probs)

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
            msg = _project_message_strength(msg, lambda_msg=lambda_msg)
            parent_nodes[j] = msg

    # Root context message.
    up_messages[0][0] = root_prior

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
                child_msg = _project_message_strength(child_msg, lambda_msg=lambda_msg)
                up_messages[depth + 1][j * s + child_pos] = child_msg

    return up_messages, down_messages



def compute_marginals(
    l: int,
    up_messages: List[np.ndarray],
    down_messages: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute node marginals by multiplying incoming messages."""
    marginals: List[np.ndarray] = []
    for depth in range(l + 1):
        prod = up_messages[depth] * down_messages[depth]
        marginals.append(_normalize_rows(prod))
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
    lambda_msg: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Run constrained-message BP for one observed leaf configuration.

    `rule_probs[level][parent, r]` gives the probability of choosing rule `r`
    for parent symbol `parent` at the selected level. If `rule_probs` is None,
    all rules are equiprobable as in the original local-budget implementation.
    """
    leaves_BP = _encode_leaves(xis, q=q, mask_symbol=mask_symbol)
    up_messages, down_messages = generate_tree(l=l, q=q, leaves=leaves_BP, num_classes=num_classes)
    up_messages, down_messages = update_messages(
        l=l,
        q=q,
        up_messages=up_messages,
        down_messages=down_messages,
        rules=rules,
        s=s,
        lambda_msg=lambda_msg,
        num_classes=num_classes,
        rule_probs=rule_probs,
        root_prior=root_prior,
    )
    marginals = compute_marginals(l=l, up_messages=up_messages, down_messages=down_messages)
    return marginals, float("nan")



def masked_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xis: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_msg: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
) -> np.ndarray:
    """Return the leaf marginals for a masked sequence."""
    marginals, _ = run_BP(
        rules=rules,
        l=l,
        q=q,
        xis=xis,
        s=s,
        num_classes=num_classes,
        lambda_msg=lambda_msg,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
    )
    return marginals[-1]



def last_token_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xi: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_msg: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
) -> np.ndarray:
    """Return the posterior of the last token after masking it."""
    xi_masked = np.asarray(xi, dtype=np.int64).copy()
    mask_symbol = q if mask_symbol is None else mask_symbol
    xi_masked[-1] = mask_symbol
    leaf_marginals = masked_inference(
        rules=rules,
        l=l,
        q=q,
        xis=xi_masked,
        s=s,
        num_classes=num_classes,
        lambda_msg=lambda_msg,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
    )
    return leaf_marginals[-1]



def run_last_token_inference(
    rules: Sequence[np.ndarray],
    l: int,
    q: int,
    xi: Sequence[int],
    s: int,
    num_classes: Optional[int] = None,
    lambda_msg: float = np.inf,
    rule_probs: Optional[Sequence[np.ndarray]] = None,
    root_prior: Optional[np.ndarray] = None,
    mask_symbol: Optional[int] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Convenience wrapper for next-token prediction.

    Returns:
        posterior, predicted_token, loss_on_true_last_token
    """
    xi = np.asarray(xi, dtype=np.int64)
    posterior = last_token_inference(
        rules=rules,
        l=l,
        q=q,
        xi=xi,
        s=s,
        num_classes=num_classes,
        lambda_msg=lambda_msg,
        rule_probs=rule_probs,
        root_prior=root_prior,
        mask_symbol=mask_symbol,
    )
    pred = int(np.argmax(posterior))
    loss = float(-np.log(np.clip(posterior[int(xi[-1])], 1e-300, 1.0)))
    return posterior, pred, loss
