import matplotlib.pyplot as plt
import numpy as np

x_ = []
y_ = []
x_l = []
y_l = []

n1 = 10
n2 = 50

for xs in np.linspace(0, 1, n1):
    for ny in np.linspace(0, 1, n2):
        x_.append(xs)
        y_.append(ny)
    x_l.append(x_)
    y_l.append(y_)
    x_ = []
    y_ = []

for ys in np.linspace(0, 1, n1):
    for nx in np.linspace(0, 1, n2):
        x_.append(nx)
        y_.append(ys)
    x_l.append(x_)
    y_l.append(y_)
    x_ = []
    y_ = []

for idx, (xl, yl) in enumerate(zip(x_l, y_l)):
    plt.plot(xl, yl, 'b', label=f'${idx}$')

plt.legend()
plt.savefig("test.png")