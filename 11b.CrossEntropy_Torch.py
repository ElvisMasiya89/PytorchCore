import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])

# y_pred has probabilities

# n_samples x n_classes = 1 x 3
y_pred_good = torch.tensor([[0.7, 0.2, 0.1]])
y_pred_bad = torch.tensor([[0.1, 0.3, 0.6]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(f'Loss1 torch: {l1:.4f}')
print(f'Loss2 torch: {l2:.4f}')

_, prediction1 = torch.max(y_pred_good, 1)
_, prediction2 = torch.max(y_pred_bad, 1)
print(prediction1)
print(prediction2)
