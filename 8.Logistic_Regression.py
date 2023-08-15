# 1) Design model (input , output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# -   forward pass: compute prediction
# -   backward pass: gradients
# -   update weights

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
n_samples, n_features = X.shape

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Train data
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

# Test data
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

input_size = n_features
output_size = 1
lr = 0.01


# Model for Logistic Regression
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()

        # define layers
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(input_size, output_size)

# loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# Training Loop
num_epochs = 100

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}:  loss = {loss.item():.8f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_classes = y_predicted.round()
    accuracy = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')



