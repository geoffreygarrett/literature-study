import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)

MAX_WIDTH = 30

# def generate_mid(width):
#     ret = []
#     for idx, nh in enumerate(range(9, MAX_WIDTH + 1)):
#         ret.append((f'z{2 + idx}', nn.Linear(7, 8 + nh)))
#         ret.append((f'a{2 + idx}', nn.LeakyReLU()))
#         ret.append((f'z{3 + idx}', nn.Linear(8 + nh, 9 + nh)))
#         ret.append((f'a{3 + idx}', nn.LeakyReLU()))
#
#         print(nh)
#
#     for idx, nh in enumerate(range(MAX_WIDTH - 9, 8)):
#         ret.append((f'z{2 + idx}', nn.Linear(7, 8 + nh)))
#         ret.append((f'a{2 + idx}', nn.LeakyReLU()))
#         ret.append((f'z{3 + idx}', nn.Linear(8 + nh, 9 + nh)))
#         ret.append((f'a{3 + idx}', nn.LeakyReLU()))
#
#         print(nh)

# [
#     ('z1', nn.Linear(7, 8)),
#     ('a1', nn.LeakyReLU())
# ]
# +


model = nn.Sequential(OrderedDict(
    [
        ('z1', nn.Linear(7, 8)),
        ('a1', nn.LeakyReLU()),
        ('z2', nn.Linear(8, 10)),
        ('a2', nn.LeakyReLU()),
        ('z3', nn.Linear(10, 15)),
        ('a3', nn.LeakyReLU()),
        ('z4', nn.Linear(15, 20)),
        ('a4', nn.LeakyReLU()),
        ('z5', nn.Linear(20, 15)),
        ('a5', nn.LeakyReLU()),
        ('z6', nn.Linear(15, 10)),
        ('a6', nn.LeakyReLU()),
        ('z7', nn.Linear(10, 8)),
        ('a7', nn.LeakyReLU()),
        ('z8', nn.Linear(8, 7)),
        ('a8', nn.LeakyReLU())
    ]
))
from sklearn.model_selection import train_test_split

model.to(device)

rec = np.load('rec.npy').reshape(-1, 7)
coe = np.load('coe.npy').reshape(-1, 7)

scaler_rec = MinMaxScaler()
scaler_coe = MinMaxScaler()

# X = scaler_rec.fit_transform(rec)
X = rec
# Y = scaler_coe.fit_transform(coe)
Y = coe

# separate test and train data randomly from X and Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=42)

X = torch.from_numpy(X_train).float().to(device)
Y = torch.from_numpy(Y_train).float().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
Y_test = torch.from_numpy(Y_test).float().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                             weight_decay=0.00005)
loss_log = []

n_epochs = 100  # or whatever
batch_size = 512  # or whatever

for epoch in range(n_epochs):

    # X is a torch Variable
    permutation = torch.randperm(X.size()[0])

    # iterate through batches
    for i in range(0, X.size()[0], batch_size):
        # get the batch
        x_batch = X[permutation[i:i + batch_size]]
        y_batch = Y[permutation[i:i + batch_size]]

        # forward pass
        y_pred = model(x_batch)

        # compute loss
        loss = loss_fn(y_pred, y_batch)
        loss_log.append(loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()
        if i % 10000 == 0:
            print(
                f'Epoch: {epoch}, Train Loss: {loss.item()}, Epoch Progress: {np.round((i / X.size()[0]) * 100, 0)}%')
        print(f'\nEpoch: {epoch} completed; Test Loss: {loss}')
        print("X",       np.round((X_test[0].reshape(1, -1).cpu().detach().numpy()), 3        ))
        print("Y_pred", np.round( (model(X_test[0].reshape(1, -1)).cpu().detach().numpy()),3  ))
        print("Y_true",  np.round((Y_test[0].reshape(1, -1).cpu().detach().numpy()) ,3        ))
        # print(f'i: {i}, Epoch: {epoch} completed; Loss: {loss}')


    # evaluate on test data and print
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, Y_test)
        print(f'\nEpoch: {epoch} completed; Test Loss: {loss}')
        print("X",       np.round(scaler_rec.inverse_transform(X_test[0].reshape(1, -1).cpu().detach().numpy()), 3        ))
        print("Y_pred", np.round( scaler_coe.inverse_transform(model(X_test[0].reshape(1, -1)).cpu().detach().numpy()),3  ))
        print("Y_true",  np.round(scaler_coe.inverse_transform(Y_test[0].reshape(1, -1).cpu().detach().numpy()) ,3        ))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch}")

# for epoch in range(epochs):
#
#     # input training example and return the prediction
#     y_pred = model.forward(Xs)
#
#     # calculate MSE loss
#     loss_cost = loss_fn(y_pred, y)  # + 0.001 * l2_loss
#     # append to loss
#     loss_log.append(loss_cost.detach())
#
#     loss = loss_cost  # + l1_loss
#     # back propagate through the loss gradients
#     loss.backward()
#
#     # update model weights
#     optimizer.step()
#
#     # remove current gradients for next iteration
#     optimizer.zero_grad()
#
#     if epoch % 500 == 0:
#         print(f'Epoch: {epoch} completed; Loss: {loss}')
