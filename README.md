# Fast Temporal Recovery

`fasttr` is an efficient package for the uniform sampling of histories of a
graph growth. Its core is in C++, but it has a python interface, thanks to
[pybind11](https://github.com/pybind/pybind11).
Sampling is also made efficient by the use of [SamplableSet](https://github.com/gstonge/SamplableSet).

The principal element of the package is the class HistorySampler, that allow
the efficient sampling of histories, but also the calculation of node marginal
arrival time, posterior distribution, etc.

## Requirements and dependencies

* A compiler with C++11 support
* `python3`
* `pybind11` version >= 2.2
* `networkx`
* `numpy`

## Installation

First, clone this repository.
```bash
```
Second, use pip to install the module.
```bash
pip install ./fasttr
```

## A peak under the hood

On the python side, one only needs to have a networkx.Graph representation of
the network, and a function specifying the attachement kernel.

On the C++ side, HistorySampler is created using a template class. This template accept any type of
node labels, if a hash object exists in the standard library. Otherwise, one needs to specialize
the std::hash structure.

```
├── src
    ├── hash_specialization.hpp
```
Then, to expose the new C++ node label to python, one needs add it to the binder.

```
├── src
    ├── bind_SamplableSetCR.hpp
```

To further wrap the new associated HistorySampler, it needs to be added to the python wrapper.

```
├── SamplableSet
    ├── _wrapper.py
```

Once this is done, the class can be used elegantly in python. Basic types are already implemented :

* `int`
* `str`

## Usage

### Initialization


```python
import networkx as nx
from fasttr import HistorySampler

G = nx.barabasi_albert_graph(100,1)

# Basic constructor calling
history_sampler = HistorySampler(G)
history_sampler = HistorySampler(G, seed=42) #uniform attachement kernel
history_sampler = HistorySampler(G, kernel=lambda k: k, seed=42) #LPA kernel
```

If you are trying to infer a certain kernel, it might be useful to specify a
gradient for the kernel according to some parameter. For instance, assuming a
kernel of the form $k + a$, where $a$ is a parameter, we would have

```python
a = 1
history_sampler = HistorySampler(G, kernel=lambda k: k + a,
                                 grad_kernel=lambda k: 1, seed=42)
```

### Basic operations : sampling, accessing members, changing the kernel

```python
#sampling histories uniformly
nb_sample = 100
history_sampler.sample(nb_sample) #the histories are stored in the object

#accessing the histories
history_list = history_sampler.get_histories()
```

**Note :** if you resample histories, the previous ones are erased.

```python
history_sampler.sample(100) #still got only 100 histories in memory
```

Other useful properties that can be accessed
```python
#log (natural base) of the probability of each history and its gradient
log_posterior_list = history_sampler.get_log_posterior()
grad_log_posterior_list = history_sampler.get_grad_log_posterior()

#mean of the marginal distribution for the arrival time of each node
marginal_mean_dict = history_sampler.get_marginal_mean()

#adjacency dict
adjacency_dict = history_sampler.get_adjacency()
```

One can change the attachement kernel function. The log posterior for each
history is then automatically recalculated.
```python
history_sampler.set_kernel(kernel=lambda k: k**0.5) #Non-linear PA kernel
```

One can also set the kernel and the gradient of the kernel at the same time.
```python
a = 1
history_sampler.set_kernel(kernel=lambda k: k**0.5 + a,
                           grad_kernel=lambda k: 1)
```
