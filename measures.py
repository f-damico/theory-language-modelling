import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def test(model, dataloader):
    """
    Test cross-entropy and accuracy of model on a dataloader.
    """
    if dataloader is None:
        return float('nan'), float('nan')

    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    if was_training:
        model.train()

    return loss / total, 1.0 * correct / total


@torch.no_grad()
def get_margin_stats(model, train_loader, max_samples=4096, batch_size=None):
    """
    Compute ordinary multiclass true-vs-best-other logit margin statistics on a
    deterministic subset of the training set.
    """
    if train_loader is None:
        return {}

    was_training = model.training
    model.eval()

    dataset = train_loader.dataset
    num_available = len(dataset)
    num_samples = min(max_samples, num_available) if max_samples is not None else num_available
    if num_samples == 0:
        return {}

    sampled_indices = list(range(num_samples))
    sampled_subset = torch.utils.data.Subset(dataset, sampled_indices)
    effective_batch_size = batch_size if batch_size is not None else train_loader.batch_size
    sampled_loader = torch.utils.data.DataLoader(
        sampled_subset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
    )

    margins = []
    for inputs, targets in sampled_loader:
        logits = model(inputs)
        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        other_logits = logits.clone()
        other_logits[torch.arange(logits.size(0), device=logits.device), targets] = float('-inf')
        max_other_logits = other_logits.max(dim=1).values
        margins.append((true_logits - max_other_logits).detach())

    if was_training:
        model.train()

    margins = torch.cat(margins, dim=0)
    return {
        'margin_min': float(margins.min().item()),
        'margin_max': float(margins.max().item()),
        'margin_mean': float(margins.mean().item()),
        'margin_std': float(margins.std(unbiased=False).item()),
        'margin_num_samples': int(margins.numel()),
    }


def get_norm_measures(model):
    """
    Compute additional norm observables when the model exposes the relevant methods.
    """
    out = {}
    if hasattr(model, 'compute_model_norm'):
        value = model.compute_model_norm()
        if torch.is_tensor(value):
            value = value.detach().item()
        out['specnorm'] = float(value)

    if hasattr(model, 'compute_model_norm_no_qk'):
        value = model.compute_model_norm_no_qk()
        if torch.is_tensor(value):
            value = value.detach().item()
        out['specnorm_no_qk'] = float(value)

    if hasattr(model, 'compute_l2_norm'):
        value = model.compute_l2_norm()
        if torch.is_tensor(value):
            value = value.detach().item()
        out['l2norm'] = float(value)

    return out


# -----------------------------------------------------------------------------
# RHM level-wise last-token margins M_l
# -----------------------------------------------------------------------------


def _unwrap_subset_indices(dataset) -> Tuple[object, torch.Tensor]:
    """
    Return the base dataset and the original integer indices represented by a
    possibly nested torch.utils.data.Subset.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base, parent_indices = _unwrap_subset_indices(dataset.dataset)
        child_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base, parent_indices[child_indices]
    return dataset, torch.arange(len(dataset), dtype=torch.long)


def _rules_to_entries(rules: Dict[int, torch.Tensor]) -> Dict[int, List[Tuple[int, Tuple[int, ...]]]]:
    """
    Convert rules[level][parent, r, :] into a list of (parent, child_tuple).
    Rules are kept in the repo's top-to-bottom order.
    """
    out = {}
    for level, rule_tensor in rules.items():
        arr = rule_tensor.detach().cpu().long()
        entries = []
        num_parents, num_rules = arr.shape[0], arr.shape[1]
        for parent in range(num_parents):
            for r in range(num_rules):
                entries.append((int(parent), tuple(int(x) for x in arr[parent, r].tolist())))
        out[int(level)] = entries
    return out


def _compatible_candidate_mask_for_level(
    sequence: Sequence[int],
    rules_entries: Dict[int, List[Tuple[int, Tuple[int, ...]]]],
    vocab_size: int,
    tuple_size: int,
    num_layers: int,
    ell: int,
) -> np.ndarray:
    """
    Compute A_ell(x) as a boolean mask over candidate last tokens.

    ell is the theory level: ell=1 is the bottom subtree of size s, ell=L is
    the whole tree.  The stored repo rules are top-to-bottom, so the first
    reduction uses rules[num_layers-1], then rules[num_layers-2], etc.
    """
    block_len = int(tuple_size ** ell)
    base_block = np.asarray(sequence[-block_len:], dtype=np.int64).copy()
    mask = np.zeros(vocab_size, dtype=bool)

    for candidate in range(vocab_size):
        block = base_block.copy()
        block[-1] = candidate

        # Each current node is represented by the set of possible symbols at
        # that node.  Leaves are observed terminals, hence singleton sets.
        current_sets: List[set[int]] = [{int(x)} for x in block]
        compatible = True

        for r in range(ell):
            rule_level = num_layers - 1 - r
            entries = rules_entries[rule_level]
            next_sets: List[set[int]] = []

            for start in range(0, len(current_sets), tuple_size):
                children = current_sets[start:start + tuple_size]
                parents = set()
                for parent, child_tuple in entries:
                    ok = True
                    for pos, child_symbol in enumerate(child_tuple):
                        if child_symbol not in children[pos]:
                            ok = False
                            break
                    if ok:
                        parents.add(parent)

                if not parents:
                    compatible = False
                    break
                next_sets.append(parents)

            if not compatible:
                break
            current_sets = next_sets

        if compatible and len(current_sets) == 1 and len(current_sets[0]) > 0:
            mask[candidate] = True

    return mask


def _available_levels_for_transformer(base_dataset, model_inputs: torch.Tensor) -> int:
    """
    Return how many RHM levels can be checked from the current transformer context.
    Higher levels are filled with NaNs if the model was trained with a shorter
    context window than the full RHM sequence.
    """
    num_layers = int(base_dataset.num_layers)
    tuple_size = int(base_dataset.tuple_size)
    if model_inputs.ndim < 3:
        return num_layers
    num_tokens = int(model_inputs.shape[1])
    available = 0
    for ell in range(1, num_layers + 1):
        if tuple_size ** ell <= num_tokens:
            available = ell
    return available


@torch.no_grad()
def get_rhm_margin_measures(
    model,
    dataloader,
    prefix: str,
    max_samples: Optional[int] = 4096,
    batch_size: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Compute empirical level-wise RHM last-token margins from model logits.

    Saved measures, one array of length L per split:
        {prefix}rhm_M_mean
        {prefix}rhm_M_pos_frac
        {prefix}rhm_survival_mean        = mean sigmoid(M_l)
        {prefix}rhm_level_penalty_mean   = mean log(1 + exp(-M_l))

    The function requires a RandomHierarchyModel dataset carrying:
        .rules, .rhm_sequences, .num_features, .tuple_size, .num_layers.
    """
    if dataloader is None:
        return {}

    base_dataset, all_indices = _unwrap_subset_indices(dataloader.dataset)

    required_attrs = ['rules', 'rhm_sequences', 'num_features', 'tuple_size', 'num_layers']
    missing = [name for name in required_attrs if not hasattr(base_dataset, name)]
    if missing:
        raise ValueError(
            'RHM margin diagnostics require the base RHM dataset to expose '
            f'{required_attrs}. Missing: {missing}. Use datasets/random_hierarchy_model.py '
            'from the modified files.'
        )

    num_available = int(all_indices.numel())
    if max_samples is not None and int(max_samples) > 0:
        num_samples = min(int(max_samples), num_available)
    else:
        num_samples = num_available

    if num_samples == 0:
        return {}

    selected = all_indices[:num_samples].long()
    effective_batch_size = int(batch_size or getattr(dataloader, 'batch_size', 1024) or 1024)

    vocab_size = int(base_dataset.num_features)
    tuple_size = int(base_dataset.tuple_size)
    num_layers = int(base_dataset.num_layers)
    rules_entries = _rules_to_entries(base_dataset.rules)

    # Accumulators ignore levels that are unavailable for shortened contexts.
    M_sum = torch.zeros(num_layers, dtype=torch.float64)
    M_pos_sum = torch.zeros(num_layers, dtype=torch.float64)
    survival_sum = torch.zeros(num_layers, dtype=torch.float64)
    penalty_sum = torch.zeros(num_layers, dtype=torch.float64)
    count = torch.zeros(num_layers, dtype=torch.float64)

    was_training = model.training
    model.eval()

    for start in range(0, num_samples, effective_batch_size):
        idx_cpu = selected[start:start + effective_batch_size]
        idx_for_features = idx_cpu.to(base_dataset.features.device)
        inputs = base_dataset.features[idx_for_features]
        targets = base_dataset.labels[idx_for_features]
        logits = model(inputs)

        # Keep logit computations on the model device; masks are moved per sample.
        available_levels = _available_levels_for_transformer(base_dataset, inputs)
        raw_sequences = base_dataset.rhm_sequences[idx_cpu].detach().cpu().long().numpy()

        for b in range(logits.shape[0]):
            z = logits[b]
            A_prev_np = np.ones(vocab_size, dtype=bool)
            for ell in range(1, num_layers + 1):
                if ell <= available_levels:
                    A_np = _compatible_candidate_mask_for_level(
                        sequence=raw_sequences[b],
                        rules_entries=rules_entries,
                        vocab_size=vocab_size,
                        tuple_size=tuple_size,
                        num_layers=num_layers,
                        ell=ell,
                    )
                    # Numerical guard: the true target must belong to A_l.  If a
                    # future dataset variant violates the exact equal-probability
                    # convention, avoid crashing but make the inconsistency visible.
                    true_y = int(raw_sequences[b, -1])
                    if not A_np[true_y]:
                        A_np[true_y] = True

                    B_np = A_prev_np & (~A_np)

                    A_mask = torch.as_tensor(A_np, dtype=torch.bool, device=z.device)
                    B_mask = torch.as_tensor(B_np, dtype=torch.bool, device=z.device)

                    logA = torch.logsumexp(z[A_mask], dim=0)
                    if bool(B_mask.any().item()):
                        logB = torch.logsumexp(z[B_mask], dim=0)
                        M = logA - logB
                    else:
                        M = torch.tensor(float('inf'), dtype=z.dtype, device=z.device)

                    # These are the four requested M_l observables.
                    M64 = M.detach().to(torch.float64).cpu()
                    M_sum[ell - 1] += M64
                    M_pos_sum[ell - 1] += float(M.item() > 0.0)
                    survival_sum[ell - 1] += torch.sigmoid(M64)
                    penalty_sum[ell - 1] += F.softplus(-M64)
                    count[ell - 1] += 1.0

                    A_prev_np = A_np
                else:
                    break

    if was_training:
        model.train()

    def _mean_or_nan(total: torch.Tensor) -> np.ndarray:
        out = torch.full_like(total, float('nan'), dtype=torch.float64)
        valid = count > 0
        out[valid] = total[valid] / count[valid]
        return out.numpy()

    return {
        f'{prefix}rhm_levels': np.arange(1, num_layers + 1, dtype=np.int64),
        f'{prefix}rhm_M_mean': _mean_or_nan(M_sum),
        f'{prefix}rhm_M_pos_frac': _mean_or_nan(M_pos_sum),
        f'{prefix}rhm_survival_mean': _mean_or_nan(survival_sum),
        f'{prefix}rhm_level_penalty_mean': _mean_or_nan(penalty_sum),
        f'{prefix}rhm_margin_num_samples': int(num_samples),
    }


def get_rhm_margin_measures_for_splits(model, train_loader, test_loader, args) -> Dict[str, object]:
    """Convenience wrapper used by init.py and main.py."""
    if not getattr(args, 'compute_rhm_margins', False):
        return {}
    if 'transformer' not in getattr(args, 'model', ''):
        raise ValueError('--compute_rhm_margins is currently implemented only for transformer models.')

    batch_size = getattr(args, 'rhm_margins_batch_size', None) or getattr(args, 'batch_size', None)
    out = {}
    out.update(
        get_rhm_margin_measures(
            model,
            train_loader,
            prefix='train_',
            max_samples=getattr(args, 'rhm_margins_max_train_samples', 4096),
            batch_size=batch_size,
        )
    )
    out.update(
        get_rhm_margin_measures(
            model,
            test_loader,
            prefix='test_',
            max_samples=getattr(args, 'rhm_margins_max_test_samples', 4096),
            batch_size=batch_size,
        )
    )
    return out
