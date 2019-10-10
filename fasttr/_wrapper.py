import networkx as nx
import numpy as np
from _fasttr import *

template_classes = {
    'int': IntHistorySampler,
    'str': StringHistorySampler
}

class HistorySampler:
    """
    This class implements a uniform sampler for histories of a graph growth.

    This class is a wrapper around a C++ implementation.
    """
    def __init__(self, G, kernel=None, grad_kernel=None, seed=None):
        """
        Creates a new HistorySampler instance.

        Args:
            G (networkx.Graph): Network structure to be used for sampling.
            kernel (function, optional): Function of the degree, attachement
                                         kernel.
            grad_kernel (function, optional): Function of the degree, gradient
                                              of kernel, according to some
                                              parameter.
            seed (int, optional): Seed used to sample elements from the set.
        """
        self.seed = seed or 42
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = lambda k: np.ones(len(k)) #uniform by default
        if grad_kernel is not None:
            self.grad_kernel = grad_kernel
        else:
            self.grad_kernel = lambda k: np.zeros(len(k)) #invariant

        #identify node type
        for n in G:
            if isinstance(n, int):
                self.cpp_type = 'int'
            elif isinstance(n, str):
                self.cpp_type = 'str'
            else:
                raise ValueError('The nodes type must be int or str')
            break

        #get kernel vector
        self.max_degree = max([k for _,k in G.degree()])
        self.degree_vector = np.array([k for k in range(self.max_degree+1)])
        kernel_vector = self.kernel(self.degree_vector)
        grad_kernel_vector = self.grad_kernel(self.degree_vector)

        # Instanciate the history sampler
        adjacency_map = {n:set(G.neighbors(n)) for n in G}
        self.history_sampler_ = template_classes[self.cpp_type](
            adjacency_map,kernel_vector,grad_kernel_vector,self.seed)
        self._wrap_methods()


    def _wrap_methods(self):
        """
        Assigns the methods of the C++ class to the wrapper.
        """
        for func_name in ['get_adjacency', 'get_rooted_adjacency',
                          'get_probability', 'get_histories',
                          'get_log_posterior', 'get_grad_log_posterior',
                          'get_ground_truth_log_posterior',
                          'get_marginal_mean', 'sample',
                          'root','unroot','set_kernel_vector',
                          'set_ground_truth']:
            setattr(self, func_name, getattr(self.history_sampler_, func_name))

    def set_kernel(self, kernel=None, grad_kernel=None):
        """
        Set a new kernel or grad kernel for the attachement process.

        Args:
            kernel (function): Function of the degree.
            grad_kernel (function, optional): Function of the degree.
        """
        if kernel is not None:
            self.kernel = kernel
        if grad_kernel is not None:
            self.grad_kernel = grad_kernel
        kernel_vector = self.kernel(self.degree_vector)
        grad_kernel_vector = self.grad_kernel(self.degree_vector)
        self.history_sampler_.set_kernel_vector(kernel_vector,
                                                grad_kernel_vector)

    def __str__(self):
        return 'HistorySampler'

    def __repr__(self):
        return str(self)
