#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the HistorySampler class of fasttr

Author: Guillaume St-Onge <guillaume.st-onge.4@ulaval.ca>
"""

import pytest
import networkx as nx
import numpy as np
from fasttr import *

def ln_history_probability(G, history, kernel):
    GR = nx.Graph()
    GR.add_edge(history[0],history[1])
    K = 2*kernel[1]
    lnp = 0.0
    for t in range(2, G.number_of_nodes()):
        i = history[t]
        GR.add_node(i)
        for j in G.neighbors(i):
            if GR.has_node(j):
                lnp += np.log(kernel[ GR.degree(j) ]) - np.log(K)
        for j in G.neighbors(i):
            if GR.has_node(j):
                K -= kernel[ GR.degree(j) ]
                GR.add_edge(i,j)
                K += kernel[ GR.degree(j) ]
        K += kernel[ GR.degree(i) ]
    return lnp

#========================
#Test
#========================

class TestInit:
    def test_int_node(self):
        edge_list = [(0,1),(1,2),(2,3)]
        G = nx.Graph(edge_list)
        H = HistorySampler(G)

    def test_str_node(self):
        edge_list = [('0','1'),('1','2'),('2','3')]
        G = nx.Graph(edge_list)
        H = HistorySampler(G)

class TestRooting:
    def test_root(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        adjacency_map = {0:{1},1:{0,2,3},2:{1},3:{1,4,5},4:{3},5:{3}}
        G = nx.Graph(edge_list)
        H = HistorySampler(G)
        H.root(0)
        rooted_adjacency = H.get_rooted_adjacency()
        expected = {0:{1},1:{2,3},3:{4,5}}
        assert expected == rooted_adjacency

    def test_unroot(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        adjacency_map = {0:{1},1:{0,2,3},2:{1},3:{1,4,5},4:{3},5:{3}}
        G = nx.Graph(edge_list)
        H = HistorySampler(G)
        H.root(0)
        H.unroot()
        rooted_adjacency = H.get_rooted_adjacency()
        assert  0 == len(rooted_adjacency)

    def test_root_nontree(self):
        edge_list = [(0,1),(1,2),(1,3),(3,4),(4,5),(2,5)]
        G = nx.Graph(edge_list)
        H = HistorySampler(G)
        H.root(0)
        rooted_adjacency = H.get_rooted_adjacency()
        expected = {0:{1},1:{2,3},2:{5},3:{4}} #bfs tree
        assert rooted_adjacency == expected

def test_source_prob():
    #barbell graph
    edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
    G = nx.Graph(edge_list)
    H = HistorySampler(G)
    probability_source = H.get_probability();
    expected = {0: 0.07142857142857142,
                3: 0.35714285714285715,
                5: 0.07142857142857142,
                4: 0.07142857142857142,
                1: 0.35714285714285715,
                2: 0.07142857142857142}
    assert probability_source == expected

class TestPosterior:
    def test_posterior_tree(self):
        #posterior in c++
        num_nodes = 100
        G = nx.barabasi_albert_graph(num_nodes,1)
        history_sampler = HistorySampler(G, seed=42)
        history_sampler.sample(10)
        gamma_list = np.linspace(0.7,1.3,10)
        log_posterior_cpp = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            log_posterior_cpp.append(
                history_sampler.get_log_posterior())
        log_posterior_cpp = np.array(log_posterior_cpp)

        #posterior in python
        H = history_sampler.get_histories()
        log_posterior_python = np.array([
            np.array([ln_history_probability(G, h, np.arange(200)**x)
                      for x in gamma_list]) for h in H])


        assert (np.abs(log_posterior_python - log_posterior_cpp.T)
                <= 10**(-12)).all()


class TestKernel:
    def test_grad_kernel(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        kernel = lambda x: lambda k: k**x
        grad_kernel = lambda x: lambda k: np.array([
            i**x*np.log(i) if i > 0 else 1. for i in k])
        epsilon = 10**(-8)
        G = nx.Graph(edge_list)
        H = HistorySampler(G, kernel=kernel(1.),
                           grad_kernel=grad_kernel(1.))
        H.sample(10)
        grad_log_posterior = np.array(H.get_grad_log_posterior())
        log_posterior_1 = np.array(H.get_log_posterior())
        #test for numerical derivative
        H.set_kernel(kernel=kernel(1.+epsilon),
                     grad_kernel=grad_kernel(1.+epsilon))
        log_posterior_2 = np.array(H.get_log_posterior())
        num_grad_log_posterior = (log_posterior_2 - log_posterior_1)/epsilon

        assert (np.abs(grad_log_posterior-num_grad_log_posterior)/\
                grad_log_posterior <= 10**(-5)).all()

class TestGroundTruth:
    def test_1(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        G = nx.Graph(edge_list)
        H = HistorySampler(G)
        H.sample(2)
        ground_truth = H.get_histories()[0]
        H.set_ground_truth(ground_truth)
        assert H.get_ground_truth_log_posterior() == H.get_log_posterior()[0]

    def test_2(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        G = nx.Graph(edge_list)
        H = HistorySampler(G)
        H.sample(2)
        ground_truth = H.get_histories()[0]
        H.set_ground_truth(ground_truth)
        kernel = lambda k: k**1.
        H.set_kernel(kernel)
        assert H.get_ground_truth_log_posterior() == H.get_log_posterior()[0]


class TestBiasSampling:
    def test_posterior_source_bias_1(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        G = nx.Graph(edge_list)

        #without bias
        history_sampler = HistorySampler(G, seed=42,source_bias=1.)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_unbiased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_unbiased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_unbiased = np.array(posterior_unbiased)

        #with bias
        history_sampler = HistorySampler(G, seed=42,source_bias=2.5)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_biased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_biased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_biased = np.array(posterior_biased)

        assert (np.abs(posterior_biased - posterior_unbiased)
                <= 10**(-4)).all()


    def test_posterior_source_bias_2(self):
        #ba graph
        num_nodes = 20
        G = nx.barabasi_albert_graph(num_nodes,1)

        #without bias
        history_sampler = HistorySampler(G, seed=42,source_bias=1.)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_unbiased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_unbiased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_unbiased = np.array(posterior_unbiased)

        #with bias
        history_sampler = HistorySampler(G, seed=42,source_bias=2.5)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_biased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_biased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_biased = np.array(posterior_biased)

        assert (np.abs(posterior_biased - posterior_unbiased)
                <= 10**(-4)).all()

    def test_posterior_sample_bias_1(self):
        #barbell graph
        edge_list = [(0,1),(1,2),(1,3),(3,4),(3,5)]
        G = nx.Graph(edge_list)

        #without bias
        history_sampler = HistorySampler(G, seed=42,sample_bias=1.)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_unbiased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_unbiased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_unbiased = np.array(posterior_unbiased)

        #with bias
        history_sampler = HistorySampler(G, seed=42,sample_bias=2.5)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_biased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_biased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_biased = np.array(posterior_biased)

        assert (np.abs(posterior_biased - posterior_unbiased)
                <= 10**(-4)).all()


    def test_posterior_sample_bias_2(self):
        #ba graph
        num_nodes = 20
        G = nx.barabasi_albert_graph(num_nodes,1)

        #without bias
        history_sampler = HistorySampler(G, seed=42,sample_bias=1.)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_unbiased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_unbiased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_unbiased = np.array(posterior_unbiased)

        #with bias
        history_sampler = HistorySampler(G, seed=42,sample_bias=2.5)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_biased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_biased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_biased = np.array(posterior_biased)

        assert (np.abs(posterior_biased - posterior_unbiased)
                <= 10**(-4)).all()

    def test_posterior_sample_and_source_bias(self):
        #ba graph
        num_nodes = 20
        G = nx.barabasi_albert_graph(num_nodes,1)

        #without bias
        history_sampler = HistorySampler(G, seed=42)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_unbiased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_unbiased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_unbiased = np.array(posterior_unbiased)

        #with bias
        history_sampler = HistorySampler(G, seed=42,source_bias=2.,sample_bias=2.5)
        history_sampler.sample(1000)
        gamma_list = np.linspace(0.7,1.3,5)
        posterior_biased = []
        for gamma in gamma_list:
            history_sampler.set_kernel(lambda k: k**(gamma))
            posterior_biased.append(
                np.mean(np.exp(history_sampler.get_log_posterior())))
        posterior_biased = np.array(posterior_biased)

        assert (np.abs(posterior_biased - posterior_unbiased)
                <= 10**(-4)).all()

