'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        x = x * self.factor
        y = y * self.factor
        return x, y


dataset = WineDataset(transform=ToTensor())

# Access the first sample in the dataset (returns a tuple with  x-features and y_label )
first_dataset = dataset[0]

# Tuple Unpack the features and label from the first sample
features, labels = first_dataset

# Print the features and label of the first sample
print("Features:", features, "Label:", labels)
print("Features Type:", type(features), "Labels:", type(labels))

print("*" * 3**5)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(5)])
dataset = WineDataset(transform=composed)
first_dataset = dataset[0]
features, labels = first_dataset
print("Features:", features, "Label:", labels)
print("Features Type:", type(features), "Labels:", type(labels))
