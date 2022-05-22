#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.ticker as plticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Palatino"],
})

mu_1 = 2
mu_2 = 0
sig_1 = 2
sig_2 = 2

def p(x):
    return norm.pdf(x, mu_2, sig_2)

def q(x):
    return norm.pdf(x, mu_1, sig_1)

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

range = np.arange(-10, 10, 0.001)

KL_int, err = quad(KL, -10, 10)
print( 'KL: ', KL_int )

fig = plt.figure(figsize=(9, 4), dpi=300)

#---------- First Plot

ax = fig.add_subplot(1,2,1)
ax.grid(True, linewidth=0.2)
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
locx = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
locy = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(locx)
plt.hlines([0], -10, 10, color="black", linewidth=0.7)
plt.vlines([0], -1, 1, color="black", linewidth=0.7)
ax.yaxis.set_major_locator(locy)
ax.grid(True, linewidth=0.1)



ax.text(-4, 0.17, r'$P\sim{}\mathcal{N}(' + f'{mu_2},{sig_2})' + '$', horizontalalignment='center',fontsize=9)
ax.text(6, 0.17, r'$Q\sim{}\mathcal{N}(' + f'{mu_1},{sig_1})' + '$', horizontalalignment='center',fontsize=9)

plt.plot(range, p(range))
plt.plot(range, q(range))

ax.set_xlim(-10,10)
ax.set_ylim(-0.1,1.3 * np.max(np.concatenate((p(range), q(range)))))

#---------- Second Plot

ax = fig.add_subplot(1,2,2)
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
locx = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
locy = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(locx)
plt.hlines([0], -10, 10, color="black", linewidth=0.7)
plt.vlines([0], -1, 1, color="black", linewidth=0.7)
ax.yaxis.set_major_locator(locy)
ax.grid(True, linewidth=0.1)
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['top'].set_color('none')
ax.set_xlim(-10,10)
ax.set_ylim(-0.1,0.25)

ax.text(3.5, 0.17, r'$D_{KL}(P\;||\;Q)=' +f'{KL_int:0.2f}' + r'$', horizontalalignment='center',fontsize=9)

ax.plot(range, KL(range), linewidth=0)

ax.fill_between(range, 0, KL(range))

plt.savefig('../graphics/kullback-leibler.pdf',bbox_inches='tight')
# plt.show()