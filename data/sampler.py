from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)
        self.ratio = len(self.data_source) // (self.num_samples * self.num_instances)

    def __iter__(self):
        ret = []
        indices = list(range(0, self.num_samples))
        for _ in range(self.ratio):
            np.random.shuffle(indices)
            for i in indices:
                pid = self.pids[i]
                t = self.index_dic[pid]
                if len(t) >= self.num_instances:
                    t = np.random.choice(t, size=self.num_instances, replace=False)
                elif len(t) > 0:
                    t = np.random.choice(t, size=self.num_instances, replace=True)
                ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.ratio * self.num_samples * self.num_instances
        