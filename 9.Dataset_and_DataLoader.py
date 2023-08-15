import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class called WineDataset
class WineDataset(Dataset):
    def __init__(self):
        # Load data from the 'wine.csv' file using numpy
        xy = np.loadtxt('./wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # Split the loaded data into input features (self.x) and target labels (self.y)
        # The features are all columns except the first one, and labels are the second column
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [1]])  # n_samples , 1

        # Store the total number of samples in the dataset
        self.n_samples = xy.shape[0]

    # Method to retrieve a sample from the dataset based on an index
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Method to return the total number of samples in the dataset
    def __len__(self):
        return self.n_samples


# Create an instance of the WineDataset class
dataset = WineDataset()

# Access the first sample in the dataset (returns a tuple with  x-features and y_label )
first_dataset = dataset[0]

# Tuple Unpack the features and label from the first sample
features, labels = first_dataset

# Print the features and label of the first sample
# print("Features:", features)
# print("Label:", labels)


# Dataloader
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
data_iter = iter(dataloader)
data = data_iter.next()
features, labels = data

print("Features:", features)
print("Label:", labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print("Total_Samples:", total_samples, "Number_of_Iterations:", n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape }')
