import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def test(model, dataloader):
    """
    Test the accuracy of model in predicting the output of data from dataloader.

    Returns:
        Classification accuracy of model in [0,1].
    """
    model.eval()

    correct = 0
    total = 0
    loss = 0.

    with torch.no_grad():
        for inputs, targets in dataloader:

            outputs = model(inputs)
            _, predictions = outputs.max(1)

            loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    return loss / total, 1.0 * correct / total


@torch.no_grad()
def get_margin_stats(model, train_loader, max_samples=4096, batch_size=None):
    """
    Compute statistics of the multiclass margin on a random subset of the training set.

    Margin for one example:
        margin = logit_true - max_{j != true} logit_j

    Args:
        model: trained model
        train_loader: training dataloader
        max_samples: maximum number of random training examples to use
        batch_size: batch size used for this measurement; if None, reuse train_loader.batch_size

    Returns:
        Dictionary with:
            margin_min
            margin_max
            margin_mean
            margin_std
            margin_num_samples
    """
    if train_loader is None:
        return {}

    model.eval()

    dataset = train_loader.dataset
    num_available = len(dataset)
    num_samples = min(max_samples, num_available)

    if num_samples == 0:
        return {}

    # Random subset of the training set
    sampled_indices = torch.randperm(num_available)[:num_samples].tolist()
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
        logits = model(inputs)  # shape: [B, C]

        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

        other_logits = logits.clone()
        other_logits[torch.arange(logits.size(0), device=logits.device), targets] = float('-inf')
        max_other_logits = other_logits.max(dim=1).values

        batch_margins = true_logits - max_other_logits
        margins.append(batch_margins.detach())

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

    Returns:
        Dictionary with spectral complexity norm, spectral complexity without Q/K,
        and l2 norm when available.
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