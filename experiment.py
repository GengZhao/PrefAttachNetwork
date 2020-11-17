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
		distribution_info = distribution+" ("+str(max_fit)+")"

	elif distribution == 'constant':
		pan = PAN(m=5)
		pan.grow_to_size(FINAL_SIZE_4)
		distribution_info = distribution
		# NOTE: DON'T CHOOSE CONSTANT! There are no valid pairs in constant model. 

	elif distribution == 'beta':
		pan = PAN(m=5, fgen=Beta(1,10))
		pan.grow_to_size(FINAL_SIZE_4)
		distribution_info = distribution

	if str(input('Number of samples for the catchup time = 100. OK? ("Y" for yes)\n')) != 'Y':
		NUM_SAMPLES_CATCHUP_TIME = int(input('How many samples?\n'))

	predictions = []
	i_s = []
	j_s = []

	for _ in range(NUM_SAMPLES_CATCHUP_TIME):

		j = np.random.randint(0, pan.size-1)
		i = np.random.randint(j+1, pan.size)

		while pan.fs[i] <= pan.fs[j]:
			j = np.random.randint(0, pan.size-1)
			i = np.random.randint(j+1, pan.size)

		i_s.append(i)
		j_s.append(j)


	if not pan.is_it_valid(i_s, j_s):
		print('ERROR! the (i_s, j_s) do not form valid pairs')

	for k in range(NUM_SAMPLES_CATCHUP_TIME):
		i = i_s[k]
		j = j_s[k]
		alpha = i/j
		beta = pan.fs[i]/pan.fs[j]
		pred = i*np.power(alpha, 1/(beta-1))
		if pred > pan.size:
			pred = pan.size+1
		predictions.append(pred)

	catchuptimes = pan.catchuptimes(i_s,j_s)
	for idx in range(len(catchuptimes)):
		if catchuptimes[idx] == None:
			# When they never catch-up, artificially set the catchup times to the size+1.
			catchuptimes[idx] = pan.size+1 

	plt.scatter(catchuptimes, predictions, marker='x')
	plt.ylabel("Math predictions (if > size, then they're set to size+1")
	plt.xlabel("Actual catch-up times (if never, then they're set to size+1)")
	plt.title('Catch-up times of '+str(NUM_SAMPLES_CATCHUP_TIME)+' random valid pairs. \n Model = '+distribution_info+', size = '+str(pan.size))
	plt.legend()
	plt.show()




