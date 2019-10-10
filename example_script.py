import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from fasttr import *

num_nodes = 10000
G = nx.barabasi_albert_graph(num_nodes,1)
print("Initializing object...")
history_sampler = HistorySampler(G, seed=42)
print("Running MC sweeps...")
history_sampler.sample(10)
print("Finished sampling")
gamma_list = np.linspace(0.9,1.1,50)
log_posterior_vector_list = []
print("Evaluating posterior for each kernel...")
for gamma in gamma_list:
    history_sampler.set_kernel(lambda k: k**(gamma))
    log_posterior_vector_list.append(history_sampler.get_log_posterior())
print("Done, plotting")
log_posterior_vector_list = np.array(log_posterior_vector_list)
log_posterior_vector_list -= np.max(log_posterior_vector_list)
posterior = np.mean(np.exp(log_posterior_vector_list), axis=1)
plt.plot(gamma_list,posterior)
plt.show()

