import torch
import torch.nn as nn
import numpy as np


# Multiclass classification

class Multiclass(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Multiclass, self).__init__()
        self.linear_layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.linear_layer2(x)
        # no softmax at the end
        return x


model = Multiclass(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # applies softmax
