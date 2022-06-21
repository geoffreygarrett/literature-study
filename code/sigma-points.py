import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"

mu_1 = 0
mu_2 = 0
sigma_1 = 1
sigma_2 = 3
covariance = np.array([[sigma_1 ** 2, 0], [0, sigma_2 ** 2]])


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


rot = rotation_matrix(-10 * np.pi / 180)
covariance = np.matmul(rot, np.matmul(covariance, rot.T))
x = np.random.multivariate_normal([mu_1, mu_2], covariance, 1500)
mu_x = np.array([mu_1, mu_2])

t = np.linspace(0, 2 * np.pi, 100)
sig_x, sig_y = mu_1 + sigma_1 * np.cos(t), mu_2 + sigma_2 * np.sin(t)
sig_x, sig_y = np.matmul(rot,  np.array([sig_x, sig_y]))


plt.figure(figsize=(4, 4), dpi=400)
plt.axis('equal')

plt.plot(sig_x, sig_y, label=r"$\sigma_{\bm{x}}$", linestyle='--', alpha=0.6)
# plt.grid(color='lightgray', linestyle='--')


# ---------------------
plt.xticks([], [])
plt.yticks([], [])
# ---------------
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
# for minor ticks
ax.set_xticks([], minor=True)
ax.set_yticks([], minor=True)
plt.scatter(x[:, 0], x[:, 1], color='gray', s=0.2, alpha=0.8, marker='.')

# plut label at mu_x
plt.text(0.2 + mu_x[0], 0.2 + mu_x[1], r"$\bm{x}$", fontsize=15)

# set x limes to +- 4
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# UKF settings
el = 2
alf = 1
kap = 0
bet = 2
lam = alf ** 2 * (el + kap) - el

L = cholesky(covariance * np.sqrt(el + lam))
# L = cholesky(covariance)

# SVD
# U, S, V = np.linalg.svd(covariance)

sigmas = [mu_x]
# weight = [lam / (el + lam) + (1 - alf ** 2 + bet)]

for i in range(el):
    sigmas.append(mu_x + L[i,:])
    sigmas.append(mu_x - L[i,:])
    # weight.append(1 / 2 / (el + lam))
    # weight.append(1 / 2 / (el + lam))

# plot L
# plt.plot(mu_1 + Lb[0, 0], mu_2 + Lb[1, 0], color='red', markersize=10)
# plt.plot(mu_1 + Lb[0, 1], mu_2 + Lb[1, 1], color='red', markersize=10)

# plot SVD vectors
# plt.plot([mu_1, mu_1 + U[0, 0] * S[0]], [mu_2, mu_2 + U[1, 0] * S[0]], color='red', markersize=10)
# plt.plot([mu_1, mu_1 + U[0, 1] * S[1]], [mu_2, mu_2 + U[1, 1] * S[1]], color='red', markersize=10)

# plot L vectors
# plt.plot([mu_1, mu_1 + L[0, 0]], [mu_2, mu_2 + Lb[1, 0]], color='red', markersize=10)
# plt.plot([mu_1, mu_1 + L[0, 1]], [mu_2, mu_2 + Lb[1, 1]], color='red', markersize=10)


# plt.hlines(mu_1, *Lb[:, 0], color='red')
# plt.vlines(mu_2, *Lb[:, 1], color='red')

# plot all sigma points

# concatenate sigma points
sigmas = np.array(sigmas)
plt.scatter(sigmas[:, 0], sigmas[:, 1], s=60, facecolors='none', edgecolors='r', label=r"$\bm{\mathcal{X}}_j$")
plt.legend()

# for weight, sigma in zip(weight, sigmas):
#     # sigma = weight * sigma
#     # print(weight)
#     plt.plot(sigma[0], sigma[1], 'x', color='red', markersize=10)


# plot ellipse representing the covariance matrix
# https://stackoverflow.com/questions/14581358/draw-a-gaussian-ellipse-using-python
# plt.show()
# tight plot
plt.bbox_inches = 'tight'

# save with transparent background
plt.savefig("sigma-points-wan.svg", bbox_inches='tight', transparent=True)

# A = np.eye(np.ones(2)) * 10
# L = np.linalg.cholesky(A)
