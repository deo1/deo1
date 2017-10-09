'''
This script performs an A/B test on two groups, where the groups are
evaluated by some metric = distinct(devices that passed) / distinct(total devices)
where `bool passed` is a binomial random variable. The outcomes of both groups
in the experiment are simulated with some noise factor in order to be able to
observe both True and False rejection of the null hypothesis (i.e. no
statistically significant difference between the control and test groups).

With reference to: https://classroom.udacity.com/courses/ud257
'''

from random import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# p_*: experiment metric: probability of binomial outcome
# as N_* gets large binomial distribution ~ normal distribution

# helper functions
p_noise = lambda p, noise: min(0.99, max(0.01, p + (random() - 0.5) * noise))
p_gen = lambda p, N: np.random.binomial(1, p, size=N)
margin_of_error = lambda z, stderr: z * stderr
confidence_interval = lambda p, moe: (p - moe, p + moe)

# simulation constants
noise = 0.1
p = random()
N = 2000
Z = {95: 1.96, 99: 2.33} # {percentile: number of std dev}
z_exp = 95 # set the confidence interval we want for this experiment

# simulate the control set
p_cont = p_noise(p, noise)
N_cont = N
stderr_cont = ((p_cont * (1 - p_cont)) / N_cont) ** (0.5)
moe_cont = margin_of_error(Z[z_exp], stderr_cont)
conf_cont = confidence_interval(p_cont, moe_cont)

# Run experiment N times of N samples
# TODO: rewrite using matrix math
set_cont = []
for _ in range(N_cont):
    set_temp = p_gen(p_cont, N_cont)
    set_cont.append(np.average(set_temp))
set_cont = np.array(set_cont)

# test set
p_test = p_noise(p, noise)
N_test = N
stderr_test = ((p_test * (1 - p_test)) / N_test) ** (0.5)
moe_test = margin_of_error(Z[z_exp], stderr_test)
conf_test = confidence_interval(p_test, moe_test)

# Run experiment N times of N samples
# TODO: rewrite using matrix math
set_test = []
for _ in range(N_test):
    set_temp = p_gen(p_test, N_test)
    set_test.append(np.average(set_temp))
set_test = np.array(set_test)

# pooled probabilities (gaussian mixture)
p_pool = ((p_cont * N_cont) + (p_test * N_test)) / (N_cont + N_test)
stderr_pool = (p_pool * (1 - p_pool) * (1/N_cont + 1/N_test)) ** 0.5
p_diff = abs(p_test - p_cont) # null hypothesis states that p_diff == 0
moe_pool = margin_of_error(Z[z_exp], stderr_pool)

# evaluate experiment results
print("\nReject null hypothesis: {}".format(p_diff > moe_pool))

# visualize results of both groups
sns.set_palette("Paired")
fig, ax = plt.subplots()
for ab in [set_cont, set_test]:
    sns.distplot(ab, ax=ax, kde=True)
 