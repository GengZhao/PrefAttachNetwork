#!/usr/bin/env python

import numpy as np
import math

# wrappers for generators
def Beta(a, b): return lambda: np.random.beta(a, b)
def Exp(s): return lambda: np.random.exponential(s)
def Poi(a): return lambda: np.random.poisson(a)
def Unif(a, b): return lambda: np.random.uniform(a, b)
def Choice(arr, p=None): return lambda: np.random.choice(arr, p=p)

class PAN(object):
    def __init__(self, m=1, fgen=lambda:1.0, capacity=1000):
        '''Preferential attachment network
        m: degree for each new node
        fgen: fitness distribution
        capacity(optional): hint for size
        edges: edges[i] is the list of m vertices that vertex m 
                attached to when it was created.
        '''
        self.capacity = capacity
        self.m = m
        self.degs = np.array([m], dtype='int32')        # degrees
        self.fgen = fgen
        self.fs = np.array([fgen()], dtype='float32')   # fitnesses
        self.edges = [[0] * m]                          # edges w/ multiplicity
        self.size = 1                                   # n nodes
        self.tot_edges = m                              # n edges
        self.weights = self.fs * self.degs
        self.tot_weights = self.weights[0]
        self._resize(capacity)

    def _resize(self, size):
        '''Pad arrays with zeros to avoid repetitive appending'''
        self.degs.resize(size)
        self.fs.resize(size)
        self.weights.resize(size)
        self.capacity = size

    def add_node(self):
        size = self.size
        m = self.m
        if size == self.capacity:
            self._resize(self.capacity * 2)
        endpoints = np.random.choice(range(size), m, replace=True, p=self.weights[:size]/self.tot_weights)
        self.edges.append(endpoints)    # new edges
        new_fit = self.fgen()                   # new fitness
        self.fs[size] = new_fit
        self.degs[size] = m                     # deg for new node
        np.add.at(self.degs, endpoints, 1)      # update the degrees (with multiplicity for multi-edges)
        self.weights[size] = m * new_fit
        endpoints_fit = self.fs[endpoints]
        np.add.at(self.weights, endpoints, endpoints_fit)
        self.tot_weights += m * new_fit + math.fsum(endpoints_fit)
        self.size += 1
        self.tot_edges += m

    def grow_to_size(self, size):
        if size > self.capacity: self._resize(size)
        while (self.size < size):
            self.add_node()
            if self.size %1000 == 0:
                print('Current size of moodel is: '+str(self.size))

    def catchuptime(self,i,j):
        """ if i>j and fitness[i] > fitness[j], this return the smallest time T such that at time T, deg[i]>=deg[j].
        If no such time T exists in the current graph, return False. If the conditions for i and j are not met, 
        return None."""
        if i<=j or self.fs[i] <= self.fs[j] or self.degs[i] < self.degs[j]:
            return None
        time = j + 1
        deg_i = 0
        deg_j = self.m
        while time < self.size and deg_i < deg_j:
            for elem in self.edges[time]:
                if elem == i:
                    deg_i += 1
                elif elem == j:
                    deg_j += 1
            time += 1
        if deg_i < deg_j:
            return False
        else:
            return time

if __name__ == '__main__':
    pan = PAN()
    pan.grow_to_size(1000)
    print(pan.degs)

