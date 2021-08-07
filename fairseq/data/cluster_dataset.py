# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from fairseq.data import data_utils
from tqdm import trange
import pickle
from . import BaseWrapperDataset

from fairseq.data import ConcatDataset, FairseqDataset
import torch


class ClusterDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens"""

    def __init__(self, dataset, vectorizer, svd, kmeans):
        super().__init__(dataset)
        self.dataset = dataset        
        self.vectorizer = vectorizer
        self.svd = svd
        self.clusterer = kmeans
                


    def __getitem__(self, index):
        item = self.dataset[index]
        # item = self.decoder.decode(self.dataset[index].numpy())
        vec = self.vectorizer.transform([self.dataset.vocab.string(item['source'])])
        vec = self.svd.transform(vec)
        cluster = self.clusterer.predict(vec)
        item['cluster'] = cluster.item()
        return item

    def __len__(self):
        return len(self.dataset)

class SingleClusterDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens"""

    def __init__(self):
        self.dataset = []
    
    def append(self, item):
        self.dataset.append(item)        

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)





def cluster_dataset(
    datasets,
    clusterer,
):
    dataset = ClusterDataset(datasets, clusterer)
    return dataset
