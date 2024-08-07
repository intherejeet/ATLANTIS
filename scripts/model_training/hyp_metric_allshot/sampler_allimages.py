import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import collections
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_labels_to_indices(labels):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices

def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace).tolist()

class UniqueClassSempler(Sampler):
    def __init__(self, labels, m_per_class, rank=0, world_size=1, seed=0):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = sorted(list(self.labels_to_indices.keys()))
        self.m_per_class = m_per_class
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        self.rotate_indices = {label: 0 for label in self.labels}

    def __len__(self):
        return (len(self.labels) // self.world_size) * self.m_per_class

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()
        size = len(self.labels) // self.world_size
        idx = idx[size * self.rank : size * (self.rank + 1)]

        for i in idx:
            label = self.labels[i]
            all_indices = self.labels_to_indices[label]
            start_idx = self.rotate_indices[label]
            end_idx = start_idx + self.m_per_class
            
            if end_idx > len(all_indices):
                if len(all_indices) >= self.m_per_class:
                    selected_indices = np.concatenate([
                        all_indices[start_idx:],
                        all_indices[:(end_idx % len(all_indices))]
                    ]).tolist()
                else:
                    selected_indices = safe_random_choice(all_indices, self.m_per_class)
                self.rotate_indices[label] = end_idx % len(all_indices)
            else:
                selected_indices = all_indices[start_idx:end_idx].tolist()
                self.rotate_indices[label] = end_idx

            idx_list += selected_indices  # Removed tolist()


        return iter(idx_list)

    def set_epoch(self, epoch):
        self.epoch = epoch
