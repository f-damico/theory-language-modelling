from itertools import product
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import dec2bin, dec2base


def sample_rules(v, n, m, s, L, seed=42):
    """
    Sample random rules for a random hierarchy model.

    Args:
        v: vocabulary size at the observable and hidden non-root levels.
        n: number of root classes.
        m: number of synonymic lower-level representations per parent symbol.
        s: tuple size / branching factor.
        L: number of hierarchy levels.
        seed: random seed for the grammar.

    Returns:
        Dictionary rules[l] with shape:
            l=0:     [n, m, s]
            l>=1:    [v, m, s]
        The order is top-to-bottom, exactly as in the original repo.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))
    rules = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)
    return rules


def sample_data_from_rules(samples, rules, n, m, s, L):
    """
    Create RHM data from sampled indices and a fixed set of grammar rules.

    Args:
        samples: tensor of data indices in [0, max_data-1].
        rules: dictionary of production rules, ordered top-to-bottom.
        n: number of root classes.
        m: number of synonymic lower-level representations.
        s: tuple size / branching factor.
        L: number of hierarchy levels.

    Returns:
        features: terminal token sequences of shape [N, s**L].
        labels: root labels of shape [N].
    """
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n

    high_level = samples.div(data_per_hl, rounding_mode='floor')
    low_level = samples % data_per_hl

    labels = high_level
    features = labels
    size = 1

    for l in range(L):
        choices = m ** size
        data_per_hl = data_per_hl // choices
        high_level = low_level.div(data_per_hl, rounding_mode='floor')
        high_level = dec2base(high_level, m, length=size).squeeze()
        features = rules[l][features, high_level]
        features = features.flatten(start_dim=1)
        size *= s
        low_level = low_level % data_per_hl

    return features, labels


def sample_data_from_labels(labels, rules, m, L):
    """
    Create RHM data by ancestral sampling with replacement.

    This avoids encoding each possible tree realisation as a single int64 sample id,
    which overflows for deep hierarchies such as L=5, s=2, m=4.

    Args:
        labels: root labels, tensor of shape [N].
        rules: dictionary of production rules, ordered top-to-bottom.
        m: number of synonymic production rules per parent.
        L: number of hierarchy levels.

    Returns:
        features: terminal token sequences of shape [N, s**L].
        labels: root labels of shape [N].
    """
    features = labels.long().reshape(-1, 1)
    for l in range(L):
        chosen_rule = torch.randint(
            low=0,
            high=m,
            size=features.shape,
            device=features.device,
        )
        features = rules[l][features, chosen_rule].flatten(start_dim=1)
    return features, labels.long()


class RandomHierarchyModel(Dataset):
    """
    Random Hierarchy Model dataset.

    The original repo only kept the final features/labels.  For the level-wise
    last-token diagnostics, we also keep the grammar and the raw terminal-token
    sequences before one-hot encoding or masking:

        self.rules
        self.sample_ids
        self.rhm_sequences
        self.rhm_root_labels

    These attributes are read by measures.get_rhm_margin_measures(...).
    """

    def __init__(
        self,
        num_features=8,
        num_classes=2,
        num_synonyms=2,
        tuple_size=2,
        num_layers=2,
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        input_format='onehot',
        whitening=0,
        replacement=None,
        transform=None,
    ):
        self.num_features = num_features
        self.num_synonyms = num_synonyms
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size
        self.input_format = input_format
        self.whitening = whitening

        rules = sample_rules(
            num_features,
            num_classes,
            num_synonyms,
            tuple_size,
            num_layers,
            seed=seed_rules,
        )
        self.rules = rules

        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1))

        # Original behaviour: sample without replacement by encoding each full
        # derivation tree as one int64 integer.  For deep hierarchies this
        # integer can exceed the int64 range even when the requested train/test
        # split is modest.  In that case we switch to ordinary ancestral
        # sampling with replacement, which is the natural i.i.d. RHM sampling
        # convention and avoids the overflow.
        int64_safe = max_data < 1e19
        if replacement is None:
            replacement = not int64_safe

        if train_size == -1 and replacement:
            raise ValueError(
                "train_size=-1 asks to enumerate the full dataset, but replacement=True "
                "uses generative i.i.d. sampling. Please set an explicit train_size."
            )

        if not replacement:
            assert int64_safe, (
                "dataset size cannot be represented with int64!! Parameters too large! "
                "Use replacement=True / generative sampling for this parameter regime."
            )
            if train_size == -1:
                samples = torch.arange(max_data)
            else:
                test_size = min(test_size, max_data - train_size)
                random.seed(seed_sample)
                samples = torch.tensor(random.sample(range(max_data), train_size + test_size))
            self.sample_ids = samples.clone()

            raw_features, raw_labels = sample_data_from_rules(
                samples,
                rules,
                num_classes,
                num_synonyms,
                tuple_size,
                num_layers,
            )
        else:
            total_size = train_size + test_size
            torch.manual_seed(seed_sample)
            labels = torch.randint(low=0, high=num_classes, size=(total_size,))
            self.sample_ids = torch.arange(total_size)
            raw_features, raw_labels = sample_data_from_labels(
                labels,
                rules,
                num_synonyms,
                num_layers,
            )

        # Full terminal sequence and root labels, kept unchanged for diagnostics.
        self.rhm_sequences = raw_features.clone().long()
        self.rhm_root_labels = raw_labels.clone().long()

        self.features = raw_features
        self.labels = raw_labels

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if 'onehot' in input_format:
            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features if 'tuples' not in input_format else num_features ** tuple_size,
            ).float()

            if whitening:
                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif 'long' in input_format:
            self.features = self.features.long() + 1
        else:
            raise ValueError(f"Unknown input_format: {input_format}")

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx: sample index.

        Returns:
            Feature-label pair at index.
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
