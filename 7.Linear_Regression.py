# 1) Design model (input , output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# -   forward pass: compute prediction
# -   backward pass: gradients
# -   update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# Reshaping the y tensor to 100x1
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1
lr = 0.01

# Model

model = nn.Linear(input_size, output_size)
# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# Training Loop

num_epochs = 100

for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f},  loss = {loss.item():.8f}')

X_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction after training: f(5)= {model(X_test).item():.3f}')

# plot

predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
