import numpy as np

"""
Implementation of offline tensor analysis from Beyond Streams and Graphs:
Dynamic Tensor Analysis.
"""
class ota2d(object):
    def __init__(self, tensor, r1 = 3, r2 = 3):
        """r1, r2 are number of PC we wish to retain"""
        self.t = tensor
        self.r1 = r1
        self.r2 = r2
        self.m, self.n, self.T = tensor.shape
        
    def fit(self, print_error = False, Us = None, Vs = None,\
            niter = 10):
        """ 
        fit the model by alternating least squares
        no need to supply Us(tart) or V start.
        niter = 10 should be sufficient.
        """
        
        # initialize U start and V start
        if Us is None or Vs is None:
            Us, d, Vs = np.linalg.svd(self.t[:,:,1])
            Us, Vs = Us[:,:self.r1], Vs[:,:self.r2]
        
        self.Ucur, self.Vcur = Us, Vs
        
        curiter = 0
        
        while curiter < niter:
            # construct the covariance for updating U
            Ucov = np.zeros([self.m, self.m])
            for i in range(self.T):
                MV = self.t[:,:,i].dot(self.Vcur)
                Ucov += MV.dot(MV.T)
            Ucov = Ucov / self.T
            eigv, Unew = np.linalg.eigh(Ucov)
            Unew = Unew[:,-self.r1:]
            
            # construct the covariance for updating V
            Vcov = np.zeros([self.n, self.n])
            for i in range(self.T):
                MtU = self.t[:,:,i].T.dot(self.Vcur)
                Vcov += MtU.dot(MtU.T)
            Vcov = Vcov / self.T
            eigv, Vnew = np.linalg.eigh(Vcov)
            Vnew = Vnew[:,-self.r2:]
            
            self.Ucur, self.Vcur = Unew, Vnew
            curiter += 1
            
            if print_error:
            # find reconstruction error, i.e. objective function value
            # eyeball this reconerror to select niter
                PU = self.Ucur.dot(self.Ucur.T)
                PV = self.Vcur.dot(self.Vcur.T)
            
                self.that = [np.linalg.multi_dot([PU, self.t[:,:,i], PV]) \
                         for i in range(self.T)]
                self.emat = [self.t[:,:,i] - self.that[i] for i \
                              in range(self.T)]
                self.reconerror = np.sum([np.linalg.norm(self.emat[i]) ** 2 \
                                          for i in range(self.T)])
                print(self.reconerror)
    
    def error(self):
        """" find reconstruction error, mean approximation error etc"""
        # projection matrices
        PU = self.Ucur.dot(self.Ucur.T)
        PV = self.Vcur.dot(self.Vcur.T)
        # fitted tensor
        self.that = [np.linalg.multi_dot([PU, self.t[:,:,i], PV]) \
                     for i in range(self.T)]
        
        # residual tensor
        self.emat = [self.t[:,:,i] - self.that[i] for i \
                      in range(self.T)]
        
        # objective function value
        self.reconerror = np.sum([np.linalg.norm(self.emat[i]) ** 2 \
                                  for i in range(self.T)])
        
        # ||S-S_hat||^2 / ||S||^2
        self.errorvec = [np.linalg.norm(self.emat[i]) ** 2 /\
                      np.linalg.norm(self.t[:,:,i]) ** 2 for \
                      i in range(self.T)]
        
        # mean of ||S-S_hat||^2 / ||S||^2 over time
        self.meanerr = np.mean(self.errorvec)
        
        return self.meanerr

#%%
if __name__=="__main__":
    """ test case """
    m, T, r = 30, 300, 3
    sigma = 0.1
    np.random.seed(5)
    Ut, R = np.linalg.qr(np.random.normal(size = (m, r)))
    Vt, R = np.linalg.qr(np.random.normal(size = (m, r)))
    M_mean = Ut.dot(np.diag([r-i for i in range(r)])).dot(Vt.T)
    Tensor = np.zeros((m, m, T))
    for i in range(T):
    	Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))        
    
    
    model = ota2d(Tensor)
    model.fit()
    model.error()
    model.meanerr
    
    
    """
    oracle error: knowing the real row/column subspace, what is recon
    error
    """
    trutherror = [np.linalg.norm(Tensor[:,:,i] - np.linalg.multi_dot([Ut, Ut.T, Tensor[:,:,i], Vt, Vt.T])) ** 2 /\
     np.linalg.norm(Tensor[:,:,i]) ** 2 \
     for i in range(T)]
    np.mean(trutherror)
