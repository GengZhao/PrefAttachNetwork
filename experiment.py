#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PrefAttachNetwork import *

plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=13)
plt.rc('figure', figsize=[8,6])

pan_plain = PAN(m=5)
pan_beta = PAN(m=5, fgen=Beta(1,10))
pan_plain.grow_to_size(100000)
pan_beta.grow_to_size(100000)
xyz=1

# Degree distribution for beta fitnesses vs flat fitness
plt.hist(np.log(pan_plain.degs), bins=50, alpha=0.8, normed=True)
plt.hist(np.log(pan_beta.degs), bins=50, alpha=0.8, normed=True)
plt.yscale('log')
plt.xlabel('log degree')
plt.legend(['Flat fitness', 'Beta(1,10) fitness'])
plt.show()

# Degree distribution within each fitness group
fitnesses = [1,2,3]
pan_discrete = PAN(fgen=Choice([1,2,3], p=[0.5, 0.3, 0.2]))
pan_discrete.grow_to_size(100000)
plt.hist([np.log(pan_discrete.degs[pan_discrete.fs==f]) for f in fitnesses], bins=20, normed=True, label=['f='+str(f) for f in fitnesses])
plt.yscale('log')
plt.xlabel('log degree')
plt.legend()
plt.show()

