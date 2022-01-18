import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import torch

print(torch.cuda.is_available())
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# dev = 'cpu'
device = torch.device(dev)

# define the training data for the XOR problem
_x = np.linspace(-1, 1, 25)
_y = np.linspace(-1, 1, 25)
_X, _Y = np.meshgrid(_x, _y)
_Z = _X ** 2 + _Y ** 2

Xs = torch.Tensor(np.concatenate([_X.reshape(-1, 1), _Y.reshape(-1, 1)], axis=1))
Xs = Xs.to(device)

y = torch.Tensor(_Z.flatten()).reshape(Xs.shape[0], 1)
y = y.to(device)

# Xs_ = []
# y_ = []
# n_samples = 20
# zero = np.linspace(0, 0.00, n_samples)
# one = np.linspace(1, 1.0, n_samples)
# for z in zero:
#     for o in one:
#         Xs_.append([z, z])
#         y_.append(0)
#
# for z in zero:
#     for o in one:
#         Xs_.append([z, o])
#         y_.append(1)
#
# for z in zero:
#     for o in one:
#         Xs_.append([o, z])
#         y_.append(1)
#
# for z in zero:
#     for o in one:
#         Xs_.append([o, o])
#         y_.append(0)
#
# Xs = torch.Tensor(Xs_)
# y = torch.Tensor(y_).reshape(Xs.shape[0], 1)

# define multi-layer perceptron model architecture
activation = {}

NH = 10

class Net(nn.Module):

    def __init__(self, nh=NH, L=2):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.z1 = nn.Linear(2, NH)
        self.z2 = nn.Linear(NH, NH)
        self.z3 = nn.Linear(NH, 1)

        self.a1 = nn.LeakyReLU()
        self.a2 = nn.LeakyReLU()
        self.a3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.z1(x)
        activation['z1'] = x

        x = self.a1(x)
        activation['a1'] = x

        x = self.z2(x)
        activation['z2'] = x

        x = self.a2(x)
        activation['z2'] = x

        x = self.z3(x)
        activation['z3'] = x

        x = self.a3(x)
        activation['a3'] = x

        return x


# model = nn.Sequential(OrderedDict([
#     ('z1', nn.Linear(2, 2)),
#     ('a1', nn.LeakyReLU()),
#     ('z2', nn.Linear(2, 1)),
#     ('a2', nn.LeakyReLU()),
# ]))
model = Net()
model.to(device)

# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#
#     return hook
#
#
# model.a1.register_forward_hook(get_activation('a1'))
# model.z2.register_forward_hook(get_activation('z2'))
# model.z1.register_forward_hook(get_activation('z1'))

# >> print(model)
# Sequential(
#   (z1): Linear(in_features=2, out_features=2, bias=True)
#   (a1): Threshold(threshold=0, value=1.0)
#   (z2): Linear(in_features=2, out_features=1, bias=True)
#   (a2): Threshold(threshold=0, value=1.0)
# )
ny = 25
nx = 25

X, Y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
X, Y = np.meshgrid(X, Y)

xx = X.reshape(-1, 1)
yy = Y.reshape(-1, 1)

gg = np.concatenate((xx, yy), axis=1)
gg_tensor = torch.tensor(gg).type(torch.float32).to(device)


epochs = 100000
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             weight_decay=0.001)
loss_log = []
l1loss_log = []

for epoch in range(epochs):

    # input training example and return the prediction
    y_pred = model.forward(Xs)

    # calculate MSE loss
    loss_cost = loss_fn(y_pred, y)  # + 0.001 * l2_loss
    # append to loss
    loss_log.append(loss_cost.detach())

    loss = loss_cost  # + l1_loss
    # back propagate through the loss gradients
    loss.backward()

    # update model weights
    optimizer.step()

    # remove current gradients for next iteration
    optimizer.zero_grad()

    # print progress
    if epoch % 50 == 0:

        output = model(gg_tensor)
        output = output.cpu().detach().numpy()
        Z = output.reshape(ny, nx)
        # Z = np.abs(output.reshape(ny, nx) - (X ** 2 + Y ** 2))

        plt.contourf(X, Y, Z, 40, cmap='RdYlGn_r')
        plt.colorbar()

        # plt.scatter([0, 1], [0, 1], c='r')
        # plt.scatter([0.5], [0.5], c='y')
        # plt.scatter([0, 1], [1, 0], c='g')

        # plt.savefig(f"output/x2y2/test_{epoch}.png")
        plt.savefig(f"output/x2y2/test_{epoch}.png")
        plt.clf()

    if epoch % 500 == 0:
        print(f'Epoch: {epoch} completed; Loss: {loss}')

# show weights and bias
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

output = model(torch.tensor([[0., 1.]]))

print(activation['a1'])

import numpy as np

# n1 = 20
# n2 = 50

# for xs in np.linspace(0, 1, n1):
#     for ny in np.linspace(0, 1, n2):
#         x_.append(xs)
#         y_.append(ny)
#     x_l.append(x_)
#     y_l.append(y_)
#     x_ = []
#     y_ = []
#
# for ys in np.linspace(0, 1, n1):
#     for nx in np.linspace(0, 1, n2):
#         x_.append(nx)
#         y_.append(ys)
#     x_l.append(x_)
#     y_l.append(y_)
#     x_ = []
#     y_ = []

ny = 25
nx = 25

X, Y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
X, Y = np.meshgrid(X, Y)

xx = X.reshape(-1, 1)
yy = Y.reshape(-1, 1)

gg = np.concatenate((xx, yy), axis=1)

output = model(torch.tensor(gg).type(torch.float32))
output = output.detach().numpy()
Z = output.reshape(ny, nx)

plt.contourf(X, Y, Z, 20, cmap='RdYlGn')
plt.colorbar()

# plt.scatter([0, 1], [0, 1], c='r')
# plt.scatter([0.5], [0.5], c='y')
# plt.scatter([0, 1], [1, 0], c='g')

plt.savefig("test.png")

# x_ = []
# y_ = []
# x_l = []
# y_l = []
#
# for xs in np.linspace(0, 1, 10):
#     for ny in np.linspace(0, 1, 50):
#         x_.append(xs)
#         y_.append(ny)
#     x_l.append(x_)
#     y_l.append(y_)
#
# for ys in np.linspace(0, 1, 10):
#     for nx in np.linspace(0, 1, 50):
#         x_.append(nx)
#         y_.append(ys)
#     x_l.append(x_)
#     y_l.append(y_)

# plt.plot(list(range(epochs)), loss_log)
# # plt.plot(list(range(epochs)), l1loss_log)
# plt.ylabel('Loss')
# plt.savefig("test.png")
