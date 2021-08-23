import torch
import numpy as np
from tqdm import trange


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices of new classes in the whole set selected for subset
        ex_indices (sequence): Indices of exemplars
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
        exemplars_transform (callable): mask new classes from exemplars
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices 
        np.random.shuffle(self.indices)
       
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        # Remove labels other then the task from target
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.indices)

