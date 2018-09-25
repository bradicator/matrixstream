#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:28:24 2018

@author: ruofeizhao

Implementation of Streaming tensor analysis from Beyond Streams and Graphs:
Dynamic Tensor Analysis.

"""
import numpy as np

class sta2d(object):
    def __init__(self, tensor, r1 = 3, r2 = 3):
        """r1, r2 are number of PC we wish to retain"""
        self.t = tensor
        self.r1 = r1
        self.r2 = r2
        self.m, self.n, self.T = tensor.shape
        
        
    def stavec(self, U, s, x, f = 1):
        """stavec tracks every column in x"""
        n = x.shape[1]
        r = U.shape[1]
        s_new = s
        for i in range(n):
            xx = x[:, i].copy() # don't let this change x and self.t! use copy!
            for j in range(r):
                y = np.dot(U[:, j], xx)
                s_new[j] = f * s_new[j] + y ** 2
                e = xx - y * U[:, j]
                U[:, j] += y / s_new[j] * e
                xx -= y * U[:, j]
                U[:, j] = U[:, j] / np.linalg.norm(U[:, j])
            U_new, R = np.linalg.qr(U)
            if not np.allclose(U_new[:,0], U[:, 0]):
                U_new[:,0] = -U_new[:,0]
        return U_new, s_new
    
    def sta(self, s, U, s2, V, x, f = 1):
        """
        sta for a new matrix: track columns for U and
        rows for V
        """
        U_new, s_new = self.stavec(U, s, x, f)
        V_new, s2_new = self.stavec(V, s2, x.T, f)
        return s_new, U_new, s2_new, V_new
    
    def fit(self, print_error = False, f = 1):
        """ fit the model according to STA rule"""
        
        # initialize U start and V start
        Us, d, Vs = np.linalg.svd(self.t[:,:,1])
        Us, Vs = Us[:,:self.r1], Vs[:,:self.r2]
        
        self.Ucur, self.Vcur = Us, Vs
        self.su, self.sv = d[:self.r1] ** 2, d[:self.r2] ** 2
        
        self.reconerror = []
        self.error = []
        
        for i in range(self.T):
            # run sta update
            X = self.t[:,:,i]
            self.su, self.Ucur, self.sv, self.Vcur = \
            self.sta(self.su, self.Ucur, self.sv, self.Vcur, X, f)
            
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
""" test case """
if __name__ == "__main__":
    m, T, r = 30, 300, 3
    sigma = 0.1
    np.random.seed(5)
    Ut, R = np.linalg.qr(np.random.normal(size = (m, r)))
    Vt, R = np.linalg.qr(np.random.normal(size = (m, r)))
    M_mean = Ut.dot(np.diag([r-i for i in range(r)])).dot(Vt.T)
    Tensor = np.zeros((m, m, T))
    for i in range(T):
    	Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))        
    
    
    model = sta2d(Tensor)
    model.fit(True)
    model.error()
    model.meanerr

