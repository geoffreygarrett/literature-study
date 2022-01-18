import numpy as np
import matplotlib.pyplot as plt


# calculate gradient of -125*(cos(x))+y^2
def gradient(x):
    _x, _y = x
    return np.array([125 * (np.sin(_x)), 2 * _y])


# calculate -125*(cos(x))+y^2
def cost(x):
    _x, _y = x
    return -125 * (np.cos(_x)) + _y ** 2


class BaseProblem:
    """
    f(x) = -125 * cos(x1) + x2^4
    """
    x0 = np.array([-np.pi * (4 / 5), 14.])
    x_1_lim = [-np.pi, np.pi]
    x_2_lim = [-5, 15]

    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.cur_iter = 0
        self.x_hist = np.zeros((max_iter, 2))
        self.f_hist = np.zeros((max_iter, 1))
        self.h = {}
        self.name = None
        self.x = self.x0

    def save_csv(self):
        # concatenate x_l and z_l
        coords = np.concatenate((self.x_hist, self.f_hist), axis=1)

        # save to csv
        np.savetxt(f"output/{self.name}{self.format_h()}.csv", coords,
                   delimiter=",",
                   fmt="%.3f")

    @property
    def x(self):
        return self.x_hist[self.cur_iter - 1]

    @x.setter
    def x(self, x):
        self.x_hist[self.cur_iter] = x
        self.f_hist[self.cur_iter] = cost(x)
        self.cur_iter += 1

    def format_h(self):
        ret = ""
        for k, v in self.h.items():
            ret = ret + f"_{k}{v}"
        return ret

    def solve(self):
        for i in range(self.max_iter - 1):
            self.step()

    def step(self):
        raise NotImplementedError("Need to implement this!")

    def save_x_plot(self):
        plt.xlim(*self.x_1_lim)
        plt.ylim(*self.x_2_lim)
        plt.plot(*self.x_hist.T)
        plt.savefig(f"output/{self.name}{self.format_h()}.png")
        plt.clf()


class VanillaGD(BaseProblem):
    def __init__(self, lr=0.01, max_iter=100):
        super().__init__(max_iter)
        self.name = "vanilla"
        self.h = {**self.h, **{"lr": lr}}

    def step(self):
        self.x = self.x - self.h["lr"] * gradient(self.x)


class MomentumGD(VanillaGD):
    def __init__(self, lr=0.01, beta=0.9, max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "momentum"
        self.h = {**self.h, **{"beta": beta}}
        self.v = np.zeros_like(self.x0)

    def step(self):
        self.v = self.h["beta"] * self.v + self.h["lr"] * gradient(self.x)
        self.x = self.x - self.v


class NesterovGD(VanillaGD):
    def __init__(self, lr=0.01, beta=0.9, max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "nesterov"
        self.h = {**self.h, **{"beta": beta}}
        self.v = np.zeros_like(self.x0)

    def step(self):
        self.v = self.h["beta"] * self.v + self.h["lr"] * gradient(
            self.x - self.h["beta"] * self.v)
        self.x = self.x - self.v


class AdagradGD(VanillaGD):
    def __init__(self, lr=0.01, epsilon=1e-8, max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "adagrad"
        self.h = {**self.h, **{"epsilon": epsilon}}
        self._g_sqr_tau = None
        self.g = None

    def step(self):
        self.g = gradient(self.x)
        self._g_sqr_tau = np.concatenate((self._g_sqr_tau, [self.g ** 2]),
                                         0) if self._g_sqr_tau is not None else np.array(
            [self.g ** 2])
        self.x = self.x - self.g * self.h["lr"] / (
            np.sqrt(np.sum(self._g_sqr_tau, axis=0) + self.h["epsilon"]))


class AdadeltaGD(VanillaGD):
    def __init__(self, lr=0.01, gamma=0.9, epsilon=1e-8, max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "adadelta"
        self.h = {**self.h, **{"epsilon": epsilon, "gamma": gamma}}
        self._g_sqr_avg = np.zeros_like(self.x0)
        # self._d_sqr_avg = np.zeros_like(self.x0)
        self._d_sqr_avg = np.square(0.001 * gradient(self.x))

    def step(self):
        self.g = gradient(self.x)
        eps = self.h["epsilon"]
        rho = self.h["gamma"]

        self._g_sqr_avg = rho * self._g_sqr_avg + (1 - rho) * np.square(self.g)

        g = (np.sqrt(self._d_sqr_avg + eps) / np.sqrt(self._g_sqr_avg + eps)
             ) * self.g

        self.x -= g
        self._d_sqr_avg = rho * self._d_sqr_avg + (1 - rho) * g * g


class RMSPropGD(VanillaGD):
    def __init__(self, lr=0.01, gamma=0.9, epsilon=1e-8, max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "rmsprop"
        self.h = {**self.h, **{"epsilon": epsilon, "gamma": gamma}}
        self._g_sqr_avg = np.zeros_like(self.x)

    def step(self):
        self.g = gradient(self.x)

        # perform rms prop update
        self._g_sqr_avg = self.h["gamma"] * self._g_sqr_avg + (
                    1 - self.h["gamma"]) * np.square(self.g)
        self.x = self.x - self.g * self.h["lr"] / (
            np.sqrt(self._g_sqr_avg + self.h["epsilon"]))


class AdamGD(VanillaGD):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.9, epsilon=1e-8,
                 max_iter=100):
        super().__init__(lr, max_iter)
        self.name = "adam"
        self.h = {**self.h,
                  **{"epsilon": epsilon, "beta1": beta1, "beta2": beta2}}
        self.m_t = np.zeros_like(self.x)
        self.v_t = np.zeros_like(self.x)

    def step(self):
        self.g = gradient(self.x)

        # perform adam update
        self.m_t = self.h["beta1"] * self.m_t + (1 - self.h["beta1"]) * self.g
        self.v_t = self.h["beta2"] * self.v_t + (
                    1 - self.h["beta2"]) * np.square(self.g)

        # bias corrected first and second moment estimates
        m_hat = self.m_t / (1 - self.h["beta1"] ** self.cur_iter)
        v_hat = self.v_t / (1 - self.h["beta2"] ** self.cur_iter)

        self.x = self.x - self.h['lr'] * m_hat / (
                    np.sqrt(v_hat) + self.h["epsilon"])

lr_l = [0.001, 0.01,0.015, 0.025, 0.1, 0.2, 0.5, 1.0]

for lr in lr_l:
    vanilla = VanillaGD(lr=lr)
    vanilla.solve()
    vanilla.save_csv()
    vanilla.save_x_plot()

for lr in lr_l:
    momentum = MomentumGD(lr=lr)
    momentum.solve()
    momentum.save_csv()
    momentum.save_x_plot()

for lr in lr_l:
    nesterov = NesterovGD(lr=lr)
    nesterov.solve()
    nesterov.save_csv()
    nesterov.save_x_plot()

for lr in lr_l:
    adagrad = AdagradGD(lr=lr)
    adagrad.solve()
    adagrad.save_csv()
    adagrad.save_x_plot()

for lr in [None]:
    adadelta = AdadeltaGD(lr=lr, max_iter=100)
    adadelta.solve()
    adadelta.save_csv()
    adadelta.save_x_plot()

for lr in lr_l:
    rmsprop = RMSPropGD(lr=lr, max_iter=100)
    rmsprop.solve()
    rmsprop.save_csv()
    rmsprop.save_x_plot()

for lr in lr_l:
    adam = AdamGD(lr=lr, max_iter=100)
    adam.solve()
    adam.save_csv()
    adam.save_x_plot()

# for lr in [0.011, 0.025, 0.040]:
#     momentum = MomentumGD(iter=100, lr=lr)
#     momentum.solve()
#     momentum.save_csv()
#     momentum.save_x_plot()

# do_vanilla(0.011, plot_xy=True)
# do_vanilla(0.025)
# do_vanilla(0.040, iter=3)
#
# do_momentum(0.011, plot_xy=True)
# do_nesterov(0.011, plot_xy=True)
# do_adagrad(0.5, plot_xy=True)
# plot x_l

#
