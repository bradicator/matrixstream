#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:38:13 2018

@author: ruofeizhao

Incremental tensor analysis in our paper
"""
import numpy as np 

class ita2d(object):
    def __init__(self, tensor, r1 = 3, r2 = 3, onemore = False):
        """r1, r2 are number of PC we wish to retain"""
        self.t = tensor
        self.r1 = r1
        self.r2 = r2
        if onemore:
            self.r1 += 1
            self.r2 += 1
        self.onemore = onemore
        self.m, self.n, self.T = tensor.shape
        
    def ita(self, first_r1_eigenvalues, first_r1_eigenvectors, first_r2_eigenvalues, 
    first_r2_eigenvectors, x, n, tol=1e-7, forget = 1):
        """
        This function does sequential estimation of the eigenvectors and eigenvalues of 
        \frac1T \sum X(:,:,t)VV^TX(:,:,t)^T and \frac1T \sum X(:,:,t)^TUU^TX(:,:,t).
        See ruofei's notes for algorithm and notation.
        :type first_k_eigenvalues: np.array
        :param first_r1_eigenvalues: the inital estimate of eignevalues, 1 by r1
        :param first_r1_eigenvectors: the initial estimate of eigenvectors, M by r1
        :param first_r2_eigenvalues: the inital estimate of eignevalues, 1 by r2
        :param first_r2_eigenvectors: the initial estimate of eigenvectors, N by r2
        :param x: the new data point, M by N
        :param n: the effective number of data points up to now, recommend capping this number
        :param tol: tolerance
        :return: Updated eigenvalues and eigenvectors.
        """
    
        # some preparation and sanity check
        r1 = len(first_r1_eigenvalues)
        r2 = len(first_r2_eigenvalues)
        f = 1 / n
        assert first_r1_eigenvectors.shape == (x.shape[0], r1), "shape of U wrong"
        assert first_r2_eigenvectors.shape == (x.shape[1], r2), "shape of V wrong"
    
        # update first_r1_eigenvalues, first_r1_eigenvectors first
        Z = np.dot(x, first_r2_eigenvectors)
        # project new data to the column space of first_r1_eigenvectors.
        ZZ = Z - first_r1_eigenvectors.dot(first_r1_eigenvectors.T).dot(Z)
    
        # only update when new data can't be explained too well by the current eigenspace.
        if np.linalg.norm(ZZ) > tol:
            Q, R = np.linalg.qr(ZZ)
            ZTU = np.dot(Z.T, first_r1_eigenvectors)
            H21 = f * np.dot(R, ZTU)
    
            # form center piece
            H = np.block([[(1-f) * forget * np.diag(first_r1_eigenvalues) + f * np.dot(ZTU.T, ZTU), H21.T],
                          [H21, f * np.dot(R, R.T)]])
    
            # perform updates
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            first_r1_eigenvalues = eigenvalues[-r1:]
            first_r1_eigenvectors = np.dot(np.hstack((first_r1_eigenvectors, Q)), eigenvectors[:, -r1:])
    
    
        # update first_r2_eigenvalues, first_r2_eigenvectors next. repeat the routine.
        # the updated first_r1_eigenvectors is used.
        Y = np.dot(x.T, first_r1_eigenvectors)
        YY = Y - first_r2_eigenvectors.dot(first_r2_eigenvectors.T).dot(Y)
        if np.linalg.norm(YY) > tol:
            Q, R = np.linalg.qr(YY)
            YTV = np.dot(Y.T, first_r2_eigenvectors)
            H21 = f * np.dot(R, YTV)
            H = np.block([[(1-f) * forget * np.diag(first_r2_eigenvalues) + f * np.dot(YTV.T, YTV), H21.T],
                          [H21, f * np.dot(R, R.T)]])
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            first_r2_eigenvalues = eigenvalues[-r2:]
            first_r2_eigenvectors = np.dot(np.hstack((first_r2_eigenvectors, Q)), eigenvectors)
            first_r2_eigenvectors = first_r2_eigenvectors[:, -r2:]
    
        return first_r1_eigenvalues, first_r1_eigenvectors, first_r2_eigenvalues,\
        first_r2_eigenvectors


    def fit(self, print_error = False, f = 1):
        """ fit the model according to ITA rule"""
        
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
            self.ita(self.su, self.Ucur, self.sv, self.Vcur, X, min(i+1, 200),forget=f)
            
            # reconstruct and find error
            PU = self.Ucur.dot(self.Ucur.T)
            PV = self.Vcur.dot(self.Vcur.T)
            
            if self.onemore:
                PU = self.Ucur[:,:-1].dot(self.Ucur[:,:-1].T)
                PV = self.Vcur[:,:-1].dot(self.Vcur[:,:-1].T)
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
    
    
    model = ita2d(Tensor)
    model.fit(True)
    model.meanerr