from torch.utils.data import Subset
import numpy as np


def uniform_sampling(num_data, buffer_size):
    indices = list(range(num_data))
    np.random.shuffle(indices)
    return indices[:buffer_size]


def create_buffer(dataset, policy, buffer_size):
    if policy == "uniform":
        indices = uniform_sampling(len(dataset), buffer_size)
        dataset = dataset.get_dataset()
    elif policy == "rare_class_sampling":
        indices = dataset.rcs_sampling(buffer_size)
    else:
        raise ValueError(f"{policy} is not a valid sampling policy")

    return Subset(dataset, indices)
