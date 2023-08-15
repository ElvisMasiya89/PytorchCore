import torch
import torch.nn as nn
import numpy as np


# Binaryclass classification

class BinaryClass(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BinaryClass, self).__init__()
        self.linear_layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        y = self.linear_layer1(x)
        y = self.relu(y)
        y = self.linear_layer2(y)
        # sigmoid at the end
        y_pred = torch.sigmoid(y)
        return y_pred


model = BinaryClass(input_size=28 * 28, hidden_size=5)
criterion = nn.BCELoss()  # applies softmax
