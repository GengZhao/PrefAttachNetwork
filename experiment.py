#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PrefAttachNetwork import *

FINAL_SIZE_1 = 20000
FINAL_SIZE_2 = 20000
FINAL_SIZE_3 = 20000
FINAL_SIZE_4 = 20000
NUM_SAMPLES_CATCHUP_TIME = 100

plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=13)
plt.rc('figure', figsize=[8,6])

change_sizes = str(input('Do you wish to change the pre-set final sizes? If yes, type "Y"; if no, type anything else.\n'))
my_experiment = str(input("What experiment do you want? \n A = histogram of degrees, flat fitness vs Beta distribution fitness \n B = histogram of degrees for fitness distribution uniform in [1,2,3] \n C = plot of catch-up times vs the theoretical predictions \n"))

if my_experiment == 'A':

	if change_sizes == 'Y':
		FINAL_SIZE_1 = int(input('Final size for plain model:\n'))
		FINAL_SIZE_2 = int(input('Final size for beta model:\n'))

	pan_plain = PAN(m=5)
	pan_beta = PAN(m=5, fgen=Beta(1,10))
	pan_plain.grow_to_size(FINAL_SIZE_1)
	pan_beta.grow_to_size(FINAL_SIZE_2)

	# Degree distribution for beta fitnesses vs flat fitness
	plt.hist(np.log(pan_plain.degs), bins=50, alpha=0.8, density=True)
	plt.hist(np.log(pan_beta.degs), bins=50, alpha=0.8, density=True)
	plt.yscale('log')
	plt.xlabel('log degree')
	plt.legend(['Flat fitness', 'Beta(1,10) fitness'])
	plt.show()

elif my_experiment == 'B':

	if change_sizes == 'Y':
		FINAL_SIZE_3 = int(input('Final size for discrete model:\n'))

	# Degree distribution within each fitness group
	fitnesses = [1,2,3]
	pan_discrete = PAN(fgen=Choice([1,2,3], p=[0.5, 0.3, 0.2]))
	pan_discrete.grow_to_size(FINAL_SIZE_3)
	plt.hist([np.log(pan_discrete.degs[pan_discrete.fs==f]) for f in fitnesses], bins=20, density=True, label=['f='+str(f) for f in fitnesses])
	plt.yscale('log')
	plt.xlabel('log degree')
	plt.legend()
	plt.show()

elif my_experiment == 'C':

	# Produce plot of catchup-times vs predictions.
	distribution = str(input('What type of fitness distribution do you want? Type "constant", "beta", or "uniform"\n'))

	if change_sizes == 'Y':
		FINAL_SIZE_4 = int(input('Final size for '+distribution+' model:\n'))

	if distribution == 'uniform': 
		max_fit = int(input("Fitness will be chosen uniformly from {1, 2, ..., n}. What is your n? (max 10)\n"))
		pan = PAN(m=5, fgen=Choice(range(1,max_fit+1,1)))
		pan.grow_to_size(FINAL_SIZE_4)

	elif distribution == 'constant':
		pan = PAN(m=5)
		pan.grow_to_size(FINAL_SIZE_4)

	elif distribution == 'beta':
		pan = PAN(m=5, fgen=Beta(1,10))
		pan.grow_to_size(FINAL_SIZE_4)

	if str(input('Number of samples for the catchup time = 100. OK? ("Y" for yes)\n')) != 'Y':
		NUM_SAMPLES_CATCHUP_TIME = int(input('How many samples?\n'))

	predictions = []
	catchuptimes = []

	for _ in range(NUM_SAMPLES_CATCHUP_TIME):

		# TO DO THIS!
		pred=0
		cutime=0

		predictions.append(pred)
		catchuptimes.append(cutime)

	pass 




