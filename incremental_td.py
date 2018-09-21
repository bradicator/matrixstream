import numpy as np

def incremental_TD(first_r1_eigenvalues, first_r1_eigenvectors, first_r2_eigenvalues, 
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




# *************************************************************************** #
# ====== This is a class that maintains the sequential estimations ========== #
# ====== for column space and row space using Gao and Zhao's TD algorithm === #
# *************************************************************************** #

class StreamTDEstimations(object):

    __slots__ = ["r1", "r2", "first_r1_eigenvectors", "first_r2_eigenvectors",\
    "first_r1_eigenvalues", "first_r2_eigenvalues", "variance_explained_bycol",\
    "variance_explained_byrow", "variance_explained_total", "effective_number_of"+\
    "_data_count"]

    def __init__(self, x, top_r1=5, top_r2=5):
        """Initializing the tracker with a small batch data"""
        x = np.array(x)
        x = x.astype(np.float)
        self.r1 = top_r1
        self.r2 = top_r2

        # deal with 2D and 3D array. initialize U, V by doing PCA
        # on vectorized tensor (row major, column major)
        try:
            self.effective_number_of_data_count = x.shape[2] 
            xbycol = np.hstack([x[:,:,i] for i in range(x.shape[2])])
            xbyrow = np.hstack([x[:,:,i].T for i in range(x.shape[2])])
        except:
            xbycol = x
            xbyrow = x.T
            self.effective_number_of_data_count = 1 

        # form U; discard eigenvectors. var_explained_bycol is defined to be
        # ||U^T X_bycol||_F^2 / ||X_bycol||_F^2.
        egnvalbycol, egnvecbycol = np.linalg.eigh(xbycol.dot(xbycol.T) / xbycol.shape[1])          
        self.first_r1_eigenvectors = egnvecbycol[:, -self.r1:]
        self.variance_explained_bycol = (np.linalg.norm(self.first_r1_eigenvectors.T.dot(xbycol))**2 /\
         np.linalg.norm(xbycol)**2)

        # same thing for forming V.
        egnvalbyrow, egnvecbyrow = np.linalg.eigh(xbyrow.dot(xbyrow.T) / xbyrow.shape[1])          
        self.first_r2_eigenvectors = egnvecbyrow[:, -self.r2:]
        self.variance_explained_byrow = (np.linalg.norm(self.first_r2_eigenvectors.T.dot(xbyrow))**2 /\
         np.linalg.norm(xbyrow)**2)
        
        # find eigenvalues of \frac1T \sum X(:,:,t)VV^TX(:,:,t)^T
        XV = xbycol.dot(self.first_r2_eigenvectors)
        egnval, egnvec = np.linalg.eigh(XV.dot(XV.T) / self.effective_number_of_data_count)
        self.first_r1_eigenvalues = egnval[-self.r1:]

        # find eigenvalues of \frac1T \sum X(:,:,t)^TUU^TX(:,:,t)
        XTU = xbyrow.dot(self.first_r1_eigenvectors)
        egnval, egnvec = np.linalg.eigh(XTU.dot(XTU.T) / self.effective_number_of_data_count)
        self.first_r2_eigenvalues = egnval[-self.r2:]

        # add one to avoid getting 0 in incremental_TD function
        self.effective_number_of_data_count += 1 

    def initial_residuals(self, x):
        """
        Calculate residuals for the initial batch data.
        Should only be called after instance initialization.
        :param x: the initial batch of data, dimension: M by N (by T optionally)
        :return: the initial batch of residuals, dimension: M by N (by T optionally)
        """
        x = np.array(x)
        x = x.astype(np.float)
        xhat = x
        
        # for 2D and 3D array case respectively
        try:
            tx = x.shape[2]
            for i in range(tx):
                xhat[:,:,i] = np.linalg.multi_dot([self.first_r1_eigenvectors, \
                    self.first_r1_eigenvectors.T, x[:,:,i], self.first_r2_eigenvectors,\
                    self.first_r2_eigenvectors.T])
        except:
            xhat = np.linalg.multi_dot([self.first_r1_eigenvectors, \
                    self.first_r1_eigenvectors.T, x, self.first_r2_eigenvectors,\
                    self.first_r2_eigenvectors.T])

        initial_batch_residuals = x - xhat

        # defined to be 1 - ||x-xhat||_F^2 / ||x||_F^2
        self.variance_explained_total = 1 - (np.linalg.norm(initial_batch_residuals)**2 /\
         np.linalg.norm(x)**2)

        return initial_batch_residuals

    def update(self, x):
        """
        updates parameter estimates of TD:
        :type x: np.array
        :param x: a new observation, M by N
        :return: residuals, M by N
        """

        # xhat_inner = U^TXV. xhat = UU^TXVV^T
        xhat_inner = np.linalg.multi_dot([self.first_r1_eigenvectors.T, x,\
         self.first_r2_eigenvectors])
        xhat = np.linalg.multi_dot([self.first_r1_eigenvectors,\
            xhat_inner, self.first_r2_eigenvectors.T])
        residuals = x - xhat


        self.variance_explained_total = 1 - (np.linalg.norm(residuals)**2 /\
         np.linalg.norm(x)**2)

        # provably, var_exp_bycol = ||U^TXV||_F^2 / ||XV||_F^2
        self.variance_explained_bycol = np.linalg.norm(xhat_inner)**2 /\
         np.linalg.norm(x.dot(self.first_r2_eigenvectors))**2
        self.variance_explained_byrow = np.linalg.norm(xhat_inner)**2 /\
         np.linalg.norm(self.first_r1_eigenvectors.T.dot(x))**2

        # apply the incremental_TD function
        self.first_r1_eigenvalues, self.first_r1_eigenvectors, self.first_r2_eigenvalues,\
        self.first_r2_eigenvectors = incremental_TD(self.first_r1_eigenvalues, self.first_r1_eigenvectors,\
        self.first_r2_eigenvalues, self.first_r2_eigenvectors, x, self.effective_number_of_data_count)

        self.effective_number_of_data_count += 1

        return residuals

if __name__ == "__main__":
    #%% test case. initialize with aa, keep seeing bb
    np.random.seed(5)
    a = np.random.normal(size=(10, 3))
    aa = a @ a.T
    egnval, egnvec = np.linalg.eigh(a @ a.T)
    b = np.random.normal(size=(10, 3))
    bb = b @ b.T
    egnval2, egnvec2 = np.linalg.eigh(bb)
    # print ground truth
    print(egnval2[7:]**2)

    print("start")
    td = StreamTDEstimations(aa, 3, 3)
    for i in range(50):
        x = bb + np.random.normal(scale = 0.1, size = (10, 10))
        res = td.update(x)
        print(td.first_r1_eigenvalues)

    #%% test case 2. initialize with aa, keep seeing linear combination of aa and bb

    # print ground truth
    print(egnval2[7:]**2)

    print("start")
    td = StreamTDEstimations(aa, 3, 3)
    for i in range(50):
        x = (50-i)/50*bb + i*aa/50
        res = td.update(x)
        print(td.first_r1_eigenvalues)
    for i in range(500):
        res = td.update(bb)
        print(td.first_r1_eigenvalues)


