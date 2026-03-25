from itertools import *
import warnings
import copy
import sys

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import dec2bin, dec2base


def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_data_from_labels( labels, rules):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and a set of class labels.

    Args:
        lables: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.randint(low=0, high=rules[l].shape[1], size=features.shape)	# Choose a random rule for each variable in the current level
        features = rules[l][features, chosen_rule].flatten(start_dim=1)			# Apply the chosen rule to each variable in the current level
    return features, labels


def sample_data_from_labels_prob( labels, rules, layer, prob):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and a set of class labels.

    Args:
        lables: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        if l==(layer-1):
            chosen_rule = torch.multinomial(prob, features.numel(), replacement=True).reshape(features.shape)
        else:
            chosen_rule = torch.randint(low=0, high=rules[l].shape[1], size=features.shape)	# Choose a random rule for each variable in the current level
        features = rules[l][features, chosen_rule].flatten(start_dim=1)			# Apply the chosen rule to each variable in the current level
    return features, labels


def sample_data_from_indices(samples, rules, v, n, m, s, L, bonus):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        bonus: Dictionary for additional output (list), includes 'noise' (randomly replace one symbol at each level), . Includes 'size' for the number of additional data. TODO: add custom positions for 'noise'

    Returns:
        A tuple containing the inputs and outputs of the model (plus additional output in bonus dict).
    """
    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n 	# div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor')	# div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl					# compute remainder (run in range(data_per_hl))

    labels = high_level		# labels are the classes (features of highest level)
    features = labels		# init input features as labels (rep. size 1)
    size = 1

    if bonus:		# extra output for additional measures
        if 'size' not in bonus.keys():
            bonus['size'] = samples.size(0)
        if 'tree' in bonus:
            tree = {}
            bonus['tree'] = tree
        if 'noise' in bonus:	# add corrupted versions of the last bonus[-1] data
            noise = {}
            noise[L] = copy.deepcopy(features[-bonus['size']:])	# copy current representation (labels)...
            noise[L][:] = torch.randint(n, (bonus['size'],))	# ...and randomly change it
            bonus['noise'] = noise
        if 'synonyms' in bonus:	# add synonymic versions of the last bonus[-1] data
            synonyms = {}
            bonus['synonyms'] = synonyms


    for l in range(L):

        choices = m**(size)
        data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

        high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
        high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

        if bonus:
            if 'tree' in bonus:
                tree[L-l] = copy.deepcopy(features[-bonus['size']:])

            if 'synonyms' in bonus:

                for ell in synonyms.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    synonyms[ell] = rules[l][synonyms[ell], high_level[-bonus['size']:]]
                    synonyms[ell] = synonyms[ell].flatten(start_dim=1)

                high_level_syn =  copy.deepcopy(high_level[-bonus['size']:]) 			# copy current representation indices...
                if l==0:
                    high_level_syn[:] = torch.randint(m, (high_level_syn.size(0),))		# ... and randomly change it (only one index at the highest level)
                else:
                    high_level_syn[:,-2] = torch.randint(m, (high_level_syn.size(0),))	# ... and randomly change the next-to-last
                synonyms[L-l] =  copy.deepcopy(features[-bonus['size']:])
                synonyms[L-l] = rules[l][synonyms[L-l], high_level_syn]
                synonyms[L-l] = synonyms[L-l].flatten(start_dim=1)
                #TODO: add custom positions for 'synonyms'
        
        features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
        size *= s								# rep. size increases by s at each level
        low_level = low_level % data_per_hl					# compute remainder (run in range(data_per_hl))

        if bonus:
            if 'noise' in bonus:

                for ell in noise.keys():	# propagate modified data down the tree TODO: randomise whole downstream propagation
                    noise[ell] = rules[l][noise[ell], high_level[-bonus['size']:]]
                    noise[ell] = noise[ell].flatten(start_dim=1)

                noise[L-l-1] =  copy.deepcopy(features[-bonus['size']:])	# copy current representation ...
                noise[L-l-1][:,-2] = torch.randint(v, (bonus['size'],))	# ... and randomly change the next-to-last feature
                #TODO: add custom positions for 'noise'


    return features, labels


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            num_classes=2,
            num_synonyms=2,
            tuple_size=2,	# size of the low-level representations
            num_layers=2,
            zipf=None,
            layer=None,
            seed_rules=0,
            seed_sample=1,
            train_size=-1,
            test_size=0,
            input_format='onehot',
            whitening=0,
            transform=None,
            replacement=False,
            bonus={}
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms 
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        self.rules = sample_rules( num_features, num_classes, num_synonyms, tuple_size, num_layers, seed=seed_rules)
 
        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1))
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > sys.maxsize and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:

            if train_size == -1:
                samples = torch.arange( max_data)

            else:
                test_size = min( test_size, max_data-train_size)
                random.seed(seed_sample)
                samples = torch.tensor( random.sample( range(max_data), train_size+test_size))

            self.features, self.labels = sample_data_from_indices(
                samples, self.rules, num_features, num_classes, num_synonyms, tuple_size, num_layers, bonus
            )

        else:

            assert not bonus, "bonus data only implemented for sampling without replacement"
            # TODO: implement bonus data for sampling with replacement
            torch.manual_seed(seed_sample)
            if train_size == -1:
                labels = torch.randint(low=0, high=num_classes, size=(max_data + test_size,))
            else:
                labels = torch.randint(low=0, high=num_classes, size=(train_size + test_size,))
            if zipf is None:
                self.features, self.labels = sample_data_from_labels(
                    labels, self.rules
                )
            else:
                assert layer is not None, "zipf law requires layer of application"
                zipf_prob=np.ones(num_synonyms)
                for i in range(num_synonyms):
                    zipf_prob[i]=(i+1)**(-1-zipf)
                self.prob=torch.from_numpy(zipf_prob/np.sum(zipf_prob))
                self.features, self.labels = sample_data_from_labels_prob(
                    labels, self.rules, layer, self.prob
                )


        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

	# TODO: implement one-hot encoding of s-tuples
        if 'onehot' in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features
            ).float()
            if bonus:
                if 'synonyms' in bonus:
                    for k in bonus['synonyms'].keys():
                        bonus['synonyms'][k] = F.one_hot(
                            bonus['synonyms'][k].long(),
                            num_classes=num_features
                        ).float()
                        bonus['synonyms'][k] = bonus['synonyms'][k].permute(0, 2, 1)
                if 'noise' in bonus:
                    for k in bonus['noise'].keys():
                        bonus['noise'][k] = F.one_hot(
                            bonus['noise'][k].long(),
                            num_classes=num_features
                        ).float()
                        bonus['noise'][k] = bonus['noise'][k].permute(0, 2, 1)

            if whitening:

                inv_sqrt_norm = (1.-1./num_features) ** -.5
                self.features = (self.features - 1./num_features) * inv_sqrt_norm
                if bonus:
                    if 'synonyms' in bonus:
                        for k in bonus['synonyms'].keys():
                            bonus['synonyms'][k] = (bonus['synonyms'][k] - 1./num_features) * inv_sqrt_norm

                    if 'noise' in bonus:
                        for k in bonus['noise'].keys():
                            bonus['noise'][k] = (bonus['noise'][k] - 1./num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif 'long' in input_format:
            self.features = self.features.long() + 1

            if bonus:
                if 'synonyms' in bonus:
                    for k in bonus['synonyms'].keys():
                        bonus['synonyms'][k] = bonus['synonyms'][k].long() + 1

                if 'noise' in bonus:
                    for k in bonus['noise'].keys():
                        bonus['noise'][k] = bonus['noise'][k].long() + 1


        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
