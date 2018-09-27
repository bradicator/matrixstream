#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:49:32 2018

@author: ruofeizhao
"""

import numpy as np 
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import mode_dot
from tensorly.tucker_tensor import tucker_to_tensor

class ita3d(object):
    def __init__(self, tensor, r1 = 3, r2 = 3, r3 = 3):
        """r1, r2, r3 are number of PC we wish to retain
        m,n,k are dimensions of the 3d tensor"""
        self.t = tensor
        self.r1, self.r2, self.r3 = r1, r2, r3
        self.m, self.n, self.k, self.T = tensor.shape
        
    def ita(self, Ulist, Dlist, mode, x, n, tol=1e-7, forget = 1):
        """
        This function updates Ulist, Dlist on a given mode for an incoming x matrix
        :param mode, along which mode are we updating
        :param Ulist, a list of 3 eigenvector matrices
        :param Dlist, a list of 3 eigenvalue array
        :param x: the new data point, M by N
        :param n: the effective number of data points up to now, recommend capping this number
        :param tol: tolerance
        :return: Updated eigenvalues and eigenvectors.
        """
        f = 1 / n
        Y = tl.tenalg.multi_mode_dot(x, [Ulist[i].T for i in range(3) if i!=mode],
                                     [i for i in range(3) if i != mode])
        
        Z = tl.unfold(Y, mode)
        
        U = Ulist[mode]
        s = Dlist[mode]
        # project new data to the column space of first_r1_eigenvectors.
        ZZ = Z - U.dot(U.T).dot(Z)
    
        # only update when new data can't be explained too well by the current eigenspace.
        if np.linalg.norm(ZZ) > tol:
            Q, R = np.linalg.qr(ZZ)
            ZTU = np.dot(Z.T, U)
            H21 = f * np.dot(R, ZTU)
    
            # form center piece
            H = np.block([[(1-f) * forget * np.diag(s) + f * np.dot(ZTU.T, ZTU), H21.T],
                          [H21, f * np.dot(R, R.T)]])
    
            # perform updates
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            Dlist[mode] = eigenvalues[-self.r1:]
            Ulist[mode] = np.dot(np.hstack((U, Q)), eigenvectors[:, -self.r1:])
    



    def fit(self, print_error = False, f = 1):
        """ fit the model according to ITA rule"""
        
        # initialize U start and V start
        _, self.Ulist = tucker(self.t[:,:,:,0], rank=[self.r1, self.r2, self.r3])
        self.Dlist = [np.zeros(self.r1), np.zeros(self.r2), np.zeros(self.r3)]

        self.reconerror = []
        self.error = []
        
        for i in range(self.T):
            # run sta update
            X = self.t[:,:,:,i]
            for j in range(3):
                self.ita(self.Ulist, self.Dlist, j, X, min(i+1, 200),forget=f)
            
            # reconstruct and find error
            temp = tl.tenalg.multi_mode_dot(X, [i.T for i in self.Ulist])
            that = tl.tenalg.multi_mode_dot(temp, [i for i in self.Ulist])
            emat = that - X
            
            self.reconerror.append(np.linalg.norm(emat) ** 2)
            self.error.append(self.reconerror[-1] / np.linalg.norm(X) ** 2)
            
            if print_error:
                print(self.reconerror[-1])
            
        self.meanerr = np.mean(self.error)

#%%
m, T, r = 10, 100, 3
sigma = 0.1
np.random.seed(5)
Tensor = np.random.normal(size = (m,m,m))
core, factors = tucker(Tensor, rank = [3, 3, 3])
Tensor = tucker_to_tensor(core, factors)
Tensor = np.outer(Tensor, np.ones(T)).reshape((m,m,m,T))
Tensor += np.random.normal(scale = sigma, size = (m,m,m, T))
Tensor[:,:,:,0] += np.random.normal(scale = 10*sigma, size = (m,m,m,))
model = ita3d(Tensor,3,3,3)
model.fit()