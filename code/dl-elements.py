import numpy as np
from orbital_expressions import *


def classical2vector(a, e, inc, raan, argp, theta, mu):
    _cos_raan = np.cos(raan)
    _sin_raan = np.sin(raan)
    _cos_argp = np.cos(argp)
    _sin_argp = np.sin(argp)
    _cos_inc = np.cos(inc)
    _sin_inc = np.sin(inc)
    _theta = theta
    _r = OrbitalExpressions().r(a * (1 - e ** 2), e, theta)  # TODO: Check
    _e = e

    _l1 = (_cos_raan * _cos_argp - _sin_raan * _sin_argp * _cos_inc)
    _l2 = (-_cos_raan * _sin_argp - _sin_raan * _cos_argp * _cos_inc)
    _m1 = (_sin_raan * _cos_argp + _cos_raan * _sin_argp * _cos_inc)
    _m2 = (-_sin_raan * _sin_argp + _cos_raan * _cos_argp * _cos_inc)
    _n1 = _sin_argp * _sin_inc
    _n2 = _cos_argp * _sin_inc

    _aux = np.array([_r * np.cos(_theta), _r * np.sin(_theta)]).T
    _transformation = np.array([[_l1, _l2],
                                [_m1, _m2],
                                [_n1, _n2]])

    # Position
    _xyz = np.matmul(_transformation, _aux)
    _x = _xyz[0]
    _y = _xyz[1]
    _z = _xyz[2]

    # Velocity
    _H = OrbitalExpressions().H(mu, a, e)
    _v_x = mu / _H * (-_l1 * np.sin(_theta) + _l2 * (_e + np.cos(_theta)))
    _v_y = mu / _H * (-_m1 * np.sin(_theta) + _m2 * (_e + np.cos(_theta)))
    _v_z = mu / _H * (-_n1 * np.sin(_theta) + _n2 * (_e + np.cos(_theta)))
    return _x, _y, _z, _v_x, _v_y, _v_z, mu


def vector2classical(r1, r2, r3, v1, v2, v3, mu):
    r = np.array([r1, r2, r3])
    v = np.array([v1, v2, v3])
    _r = np.linalg.norm(r)
    _v = np.linalg.norm(v)
    _h = np.cross(r, v)
    _N = np.cross(np.array([0, 0, 1]).T, _h)
    _N_xy = np.sqrt(_N[0] ** 2 + _N[1] ** 2)
    _a = OrbitalExpressions().a(_r, _v, mu)
    _e_vec = (np.cross(v, _h) / mu) - (r / _r)
    _e = np.linalg.norm(_e_vec)
    _inc = np.arccos(_h[-1] / np.linalg.norm(_h))
    _raan = np.arctan2(_N[1] / _N_xy, _N[0] / _N_xy)

    _s1 = 1 if np.dot(np.cross(_N / np.linalg.norm(_N), _e_vec), _h) > 0 else -1
    _argp = _s1 * np.arccos(
        np.dot(_e_vec / np.linalg.norm(_e_vec),
               _N / np.linalg.norm(_N))
    )

    _s2 = 1 if np.dot(np.cross(_e_vec, r), _h) > 0 else -1
    _theta = _s2 * np.arccos(
        np.dot(
            r / np.linalg.norm(r),
            _e_vec / np.linalg.norm(_e_vec))
    )

    _theta = 2 * np.pi + _theta if _theta <= 0 else _theta
    _raan = 2 * np.pi + _raan if _raan <= 0 else _raan
    _argp = 2 * np.pi + _argp if _argp <= 0 else _argp

    return _a, _e, _inc, _raan, _argp, _theta


ns = 2

sma_s = np.linspace(0.1, 15 * 1.5e8, ns)
ecc_s = np.linspace(0, 0.9, ns)
inc_s = np.linspace(0, np.pi, ns)
raan_s = np.linspace(0, 2 * np.pi, ns)
argp_s = np.linspace(0, 2 * np.pi, ns)
theta_s = np.linspace(0, 2 * np.pi, ns)
mu_s = np.linspace(7.329e10, 1.3e20, ns)

SMA, ECC, INC, RAAN, ARGP, THETA, MU = np.meshgrid(sma_s, ecc_s, inc_s, raan_s,
                                                   argp_s, theta_s, mu_s)

coe = np.concatenate((
    SMA.reshape(1, -1),
    ECC.reshape(1, -1),
    INC.reshape(1, -1),
    RAAN.reshape(1, -1),
    ARGP.reshape(1, -1),
    THETA.reshape(1, -1),
    MU.reshape(1, -1)
), axis=0)

# np.save('coe.npy', coe)

r1_, r2_, r3_, v1_, v2_, v3_, mu_ = [], [], [], [], [], [], []

for sma, ecc, inc, raan, argp, theta, mu in coe.T:
    r1, r2, r3, v1, v2, v3, mu = classical2vector(
        sma, ecc, inc, raan, argp, theta, mu)

    r1_.append(r1)
    r2_.append(r2)
    r3_.append(r3)
    v1_.append(v1)
    v2_.append(v2)
    v3_.append(v3)
    mu_.append(mu)

rec = np.concatenate((
    np.array([r1_]).reshape(1, -1),
    np.array([r2_]).reshape(1, -1),
    np.array([r3_]).reshape(1, -1),
    np.array([v1_]).reshape(1, -1),
    np.array([v2_]).reshape(1, -1),
    np.array([v3_]).reshape(1, -1),
    np.array([mu_]).reshape(1, -1)
), axis=0)

print(rec.T[0])
print(coe.T[0])
# np.save('rec.npy', rec)


# r = vec_f(SMA, ECC, INC, RAAN, ARGP, THETA, MU)

# print(r)
# print(SMA.shape)
