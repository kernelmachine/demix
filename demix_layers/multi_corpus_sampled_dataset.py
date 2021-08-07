# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List
import torch
import pickle

import numpy as np

from . import FairseqDataset

def uniform_sampler(x):
    # Sample from uniform distribution
    return np.random.choice(x, 1).item()

def weighted_sampler(x, p):
    # Sample from weighted distribution
    return np.random.choice(x, 1, p=p).item()


class MultiCorpusSampledDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together and in every iteration
    creates a batch by first sampling a dataset according to a specified
    probability distribution and then getting instances from that dataset.

    Args:
        datasets: an OrderedDict of FairseqDataset instances.
        sampling_func: A function for sampling over list of dataset keys.
            The default strategy is to sample uniformly.
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        dataset_lens: List[int],
        gpu_mappings: Dict[int, int] = None,
        sampling_func: Callable[[List], int] = None,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        self.dataset_lens = dataset_lens
        self.gpu_mappings = gpu_mappings
        if sampling_func is None:
            sampling_func = uniform_sampler
        self.sampling_func = sampling_func

        self._ordered_indices = None
        if not self.gpu_mappings:
            gpus  = range(torch.distributed.get_world_size())
            if len(gpus) >= 8:
                num_gpus_per_domain = torch.distributed.get_world_size() // 8
            else:
                num_gpus_per_domain = 1
            gpu_mappings = [list(gpus[n:n+num_gpus_per_domain]) for n in range(0, len(gpus), num_gpus_per_domain)]
            self.mappings = {}
            for ix, gpus in enumerate(gpu_mappings):
                for gpu in gpus:
                    self.mappings[gpu] = ix
        else:
            self.mappings = self.gpu_mappings

    def __len__(self):
        """
        Length of this dataset is the maximum of individual datasets
        """
        return max(self.dataset_lens)

    def ordered_indices(self):
        """
        Ordered indices for batching. Here we call the underlying
        dataset's ordered_indices() so that we get the same random ordering
        as we would have from using the underlying dataset directly.
        """
        if self._ordered_indices is None:
            datasets = list(self.datasets.keys())
            if len(datasets) > 1 and torch.distributed.is_initialized():
                selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
            else:
                selected_key = datasets[0]
            dataset = self.datasets[selected_key]
            self._ordered_indices = dataset.ordered_indices()

        return np.arange(len(self))

    def _map_index_to_dataset(self, key: int, index: int):
        """
        Different underlying datasets have different lengths. In order to ensure
        we are not accessing an index outside the range of the current dataset
        size, we wrap around. This function should be called after we have
        created an ordering for this and all underlying datasets.
        """
        assert (
            self._ordered_indices is not None
        ), "Must call MultiCorpusSampledDataset.ordered_indices() first"
        mapped_index = index % len(self.datasets[key])
        return self._ordered_indices[mapped_index]

    def __getitem__(self, index: int):
        """
        Get the item associated with index from each underlying dataset.
        Since index is in the range of [0, TotalNumInstances], we need to
        map the index to the dataset before retrieving the item.
        """
        datasets = list(self.datasets.keys())
        if len(datasets) > 1 and torch.distributed.is_initialized():
            selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
        else:
            selected_key = datasets[0]
        dataset = self.datasets[selected_key]
        return dataset[self._map_index_to_dataset(selected_key, index)]


    def collater(self, samples: List[Dict]):
        """
        Generate a mini-batch for this dataset.
        To convert this into a regular mini-batch we use the following
        logic:
            1. Select a dataset using the specified probability distribution.
            2. Call the collater function of the selected dataset.
        """
        if len(samples) == 0:
            return None
        datasets = list(self.datasets.keys())
        if len(datasets) > 1 and torch.distributed.is_initialized():
            selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
        else:
            selected_key = datasets[0]
        return self.datasets[selected_key].collater(samples)

    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. Here
        we return the max across all examples at index across all underlying
        datasets.
        """
        datasets = list(self.datasets.keys())
        if len(datasets) > 1 and torch.distributed.is_initialized():
            selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
        else:
            selected_key = datasets[0]
        dataset = self.datasets[selected_key]
        return dataset.num_tokens(self._map_index_to_dataset(selected_key, index))

    def size(self, index: int):
        """
        Return an example's size as a float or tuple. Here we return the max
        across all underlying datasets. This value is used when filtering a
        dataset with max-positions.
        """
        datasets = list(self.datasets.keys())
        if len(datasets) > 1 and torch.distributed.is_initialized():
            selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
        else:
            selected_key = datasets[0]
        dataset = self.datasets[selected_key]
        return dataset.size(self._map_index_to_dataset(selected_key, index))
    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        datasets = list(self.datasets.keys())
        if len(datasets) > 1 and torch.distributed.is_initialized():
            selected_key = datasets[self.mappings[torch.distributed.get_rank(torch.distributed.group.WORLD)]]
        else:
            selected_key = datasets[0]
        dataset = self.datasets[selected_key]
        dataset.prefetch(
            [self._map_index_to_dataset(selected_key, index) for index in indices]
        )

    @property
    def supports_fetch_outside_dataloader(self):
        return all(
            self.datasets[key].supports_fetch_outside_dataloader
            for key in self.datasets
        )
