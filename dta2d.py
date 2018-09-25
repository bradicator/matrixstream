#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:06:16 2018

@author: ruofeizhao

Implementation of Dynamic tensor analysis from Beyond Streams and Graphs:
Dynamic Tensor Analysis.
"""

import numpy as np

class dta2d(object):
    def __init__(self, tensor, r1 = 3, r2 = 3):
        """r1, r2 are number of PC we wish to retain"""
        self.t = tensor
        self.r1 = r1
        self.r2 = r2
        self.m, self.n, self.T = tensor.shape
        
        
        
    def fit(self, print_error = False):
        """ fit the model according to DTA rule"""
        
        # initialize U start and V start
        
        self.Ucur = np.zeros([self.m, self.r1])
        self.Vcur = np.zeros([self.n, self.r2])
        self.Uegv, self.Vegv = np.zeros(self.r1), np.zeros(self.r2)
        
        self.reconerror = []
        self.error = []
        
        for i in range(self.T):
            # construct the covariance for updating U
            X = self.t[:,:,i]
            Ucov = self.Ucur.dot(np.diag(self.Uegv)).dot(self.Ucur.T)+X.dot(X.T)  
            eigv, Unew = np.linalg.eigh(Ucov)
            self.Ucur = Unew[:,-self.r1:]
            self.Uegv = eigv[-self.r1:]
            
            # construct the covariance for updating V
            Vcov = self.Vcur.dot(np.diag(self.Vegv)).dot(self.Vcur.T)+X.T.dot(X)  
            eigv, Vnew = np.linalg.eigh(Vcov)
            self.Vcur = Vnew[:,-self.r2:]
            self.Vegv = eigv[-self.r2:]
            
            # reconstruct and find error
            PU = self.Ucur.dot(self.Ucur.T)
            PV = self.Vcur.dot(self.Vcur.T)
            
            that = PU.dot(X).dot(PV)
            emat = that - X
            
            self.reconerror.append(np.linalg.norm(emat) ** 2)
            self.error.append(self.reconerror[-1] / np.linalg.norm(X) ** 2)
            
            if print_error:
                print(self.reconerror[-1])
            
            self.meanerr = np.mean(self.error)

#%%
if __name__ == "__main__":
    """ test case """
    # gen data
    m, T, r = 30, 300, 3
    sigma = 0.1
    np.random.seed(5)
    Ut, R = np.linalg.qr(np.random.normal(size = (m, r)))
    Vt, R = np.linalg.qr(np.random.normal(size = (m, r)))
    M_mean = Ut.dot(np.diag([r-i for i in range(r)])).dot(Vt.T)
    Tensor = np.zeros((m, m, T))
    for i in range(T):
    	Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))        
    
    # fit model
    model = dta2d(Tensor)
    model.fit(True)
