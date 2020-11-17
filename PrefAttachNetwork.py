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
                print('Current size of model is: '+str(self.size))

    def is_it_valid(self, i_s, j_s):
        """ A pair (i,j) is valid if i>j and fitness[i]>fitness[j], i.e. if we can compute the catchup times. 
        Given an array of i's and an array of j's, this returns False if some pair i_s[n] and j_s[n] isn't valid."""
        if len(i_s) != len(j_s):
            return False
        for n in range(len(i_s)):
            if i_s[n] <= j_s[n] or self.fs[i_s[n]] <= self.fs[j_s[n]]:
                return False
        return True

    def catchuptimes(self,i_s,j_s):
        """ Return an array where the n-th entry is the catchup time of i_s[n] to j_s[n]. This presupposes that 
        the input is valid (see above function). If the pair doesn't catchup on time, we put None instead."""
        n = len(i_s)
        ans = [None]*n
        deg_i = {}
        for i in i_s:
            deg_i[i] = 0
        deg_j = {}
        for j in j_s:
            deg_j[j] = 0
        time = -1
        finished_pairs = 0

        while time<self.size-1:
            time += 1

            if time in i_s:
                deg_i[time] += self.m
            if time in j_s:
                deg_j[time] += self.m

            for neighbour in self.edges[time]:
                if neighbour != time: # don't count self-loops twice
                    if neighbour in i_s:
                        deg_i[neighbour] += 1
                    if neighbour in j_s:
                        deg_j[neighbour] += 1

            # Check if our pairs have catched up.            
            for k in range(n):
                if ans[k] == None and time >= i_s[k] and deg_i[i_s[k]] >= deg_j[j_s[k]]:
                    ans[k] = time
                    finished_pairs += 1

            if finished_pairs == n:
                break

            if time%1000==0:
                print("At stage "+str(time)+"/"+str(self.size)+" of computing catch-up times.")

        return ans 

if __name__ == '__main__':
    pan = PAN()
    pan.grow_to_size(1000)
    print(pan.degs)

