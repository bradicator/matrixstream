#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ita2d import ita2d
from sta2d import sta2d
from dta2d import dta2d
import numpy as np
import time
import pandas as pd


# In[2]:


# specify the m and r for four settings here.
setting_list = [(100, 5), (100, 10), (200, 5), (200, 10), (400, 5), (400, 10), (800, 5), (800, 10),                 (1600, 5), (1600, 10), (3200, 5), (3200, 10)]


# In[3]:


df = pd.DataFrame(index=["ita", "sta", "dta", "truth", "ita t", " sta t", "dta t"],                   columns = [str(i) for i in setting_list])


# In[7]:


# some shared global parameters
nrep = 30
sigma = 0.0001
T = 100


# In[18]:


# loop thru all settings
for m, r in setting_list:

    # generate mean
    np.random.seed(5)
    Ut, R = np.linalg.qr(np.random.normal(size = (m, 2*r)))
    Vt, R = np.linalg.qr(np.random.normal(size = (m, 2*r)))
    Ut1, Ut2 = Ut[:,:r], Ut[:,r:]
    Vt1, Vt2 = Vt[:,:r], Vt[:,r:]


    M_mean1 = Ut1.dot(np.diag([r-i for i in range(r)])).dot(Vt1.T)
    M_mean2 = Ut2.dot(np.diag([r-i for i in range(r)])).dot(Vt2.T)

    # container for error and exec time
    ita_e, sta_e, dta_e = [], [], []
    ita_t, sta_t, dta_t = [], [], []
    truth_e = []

    for i in range(nrep):
        # generate tensor
        Tensor = np.zeros((m, m, T))
        for t in range(T):
            Tensor[:,:,t] = (T-t)/T*M_mean1 + t/T*M_mean2 + np.random.normal(scale = sigma, size = (m, m))

        # truth error
        trutherror = []
        for t in range(T):
            trueM = (T-t)/T*M_mean1 + t/T*M_mean2
            uu, _, vv = np.linalg.svd(trueM) # note svd yields u, d, v.T
            uu, vv = uu[:,:r], vv[:r,:]
            trutherror.append(np.linalg.norm(Tensor[:,:,t] - np.linalg.multi_dot([uu, uu.T, Tensor[:,:,t], vv.T, vv]))             ** 2 / np.linalg.norm(Tensor[:,:,t]) ** 2)
        truth_e.append(np.mean(trutherror))

        # ita
        starttime = time.time()
        model = ita2d(Tensor, r, r)
        model.fit(f=0.9)
        ita_e.append(model.meanerr)
        ita_t.append(time.time()-starttime)

        # dta
        starttime = time.time()
        model = dta2d(Tensor, r, r)
        model.fit()
        dta_e.append(model.meanerr)
        dta_t.append(time.time()-starttime)

        # sta. if m > 1000, too time consuming, skip
        if m < 10:
            starttime = time.time()
            model = sta2d(Tensor, r, r)
            model.fit()
            sta_e.append(model.meanerr)
            sta_t.append(time.time()-starttime)


    df[str((m, r))] = [np.mean(i) for i in [ita_e, sta_e, dta_e, truth_e, ita_t, sta_t, dta_t]]
    
    # just save after each setting in case of timeout and loss of all results
    df.to_csv("exp3.csv")

