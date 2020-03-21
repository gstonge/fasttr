import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from fasttr import *

#parameter
num_nodes = 10000
n = 10
gamma_list = np.linspace(0.9,1.1,50) #for posterior evaluation
var_gamma = gamma_list[1]-gamma_list[0]
G = nx.barabasi_albert_graph(num_nodes,1)

#sample and evaluate posterior
print("Initializing object...")
history_sampler = HistorySampler(G, seed=42, sample_bias=2.)
print("Running MC sweeps...")
history_sampler.sample(n)
print("Finished sampling")
log_posterior_array = []
print("Evaluating posterior for each kernel...")
for gamma in gamma_list:
    history_sampler.set_kernel(lambda k: k**(gamma))
    log_posterior_array.append(history_sampler.get_log_posterior())
print("Done!")
log_posterior_array = np.array(log_posterior_array)
log_posterior_array -= np.max(log_posterior_array)
posterior = np.mean(np.exp(log_posterior_array), axis=1)
posterior /= var_gamma*np.sum(posterior)

#plot the posterior distribution
plt.plot(gamma_list,posterior)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$P(\gamma|G)$")
plt.show()

