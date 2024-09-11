import numpy as np
import os

import torch

from torch.utils import data

class Parser(data.Dataset):
    def __init__(self, config: Config, mode='train', use_cuda=False):
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid mode: {mode}')
        self.nodes = np.load(os.path.join(config.dataset_dir, f'{config.dataset_name}_loc_filled.npy'))
        self.feature = np.load(os.path.join(config.dataset_dir, f'{config.dataset_name}_feature.npy'))
        self.pm25 = np.load(os.path.join(config.dataset_dir, f'{config.dataset_name}_pm25.npy'))
        self.node_num = self.nodes.shape[0]
        self.K = config.K
        self.use_cuda = use_cuda
        # subgraph
        if config.ablation == "SG":
            edges = np.load(
                os.path.join(config.dataset_dir, f'{config.dataset_name}_edge_index.npy'))
            adj_matrix = np.zeros((30, 30), dtype=int)
            for edge in edges:
                adj_matrix[edge[0], edge[1]] = 1
            self.subgraph_w = np.tile(adj_matrix[np.newaxis, ...], (self.pm25.shape[0], 1, 1))
            self.subgraph_idx = self.subgraph_w
        else:
            self.subgraph_idx = np.load(
                os.path.join(config.dataset_dir, f'{config.dataset_name}_{self.K}_subgraph.npy'))
            self.subgraph_w = np.load(
                os.path.join(config.dataset_dir, f'{config.dataset_name}_{self.K}_subgraph_w.npy'))

        pass

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.locs[index], self.embedding_feature[index]