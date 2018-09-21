from incremental_td import incremental_TD
import numpy as np
import time

nrep = 1

# gen data from mech1
m, T, r = 100, 100, 10
sigma = 1
np.random.seed(5)
Ut, R = np.linalg.qr(np.random.normal(size = (m, r)))
Vt, R = np.linalg.qr(np.random.normal(size = (m, r)))
M_mean = Ut.dot(np.diag([r-i for i in range(r)])).dot(Vt.T)
Tensor = np.zeros((m, m, T))
for i in range(T):
	Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))

# prepare starting points for DTA and TD
# reset r here
r = 5
Ut = Ut[:,:r]
Vt = Vt[:,:r]
Us, d, Vs = np.linalg.svd(Tensor[:,:,1])
Us = Us[:,:r]
Vs = Vs[:,:r]



#####################################################################
###   ITD                                                         ###
###   ITD                                                         ###
###   ITD                                                         ###
#####################################################################

time_record = np.zeros(nrep)
mean_acc = np.zeros(nrep)

for rep in range(nrep):
	# holder for accuracy by T
	acc = np.zeros(T)

	# regenerate data
	for i in range(T):
		Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))
	
	# set starting point
	U, V = Us, Vs
	s1 = s2 = np.ones(r)

	start_time = time.time()
	for i in range(T):
		s1, U, s2, V = incremental_TD(s1, U, s2, V, Tensor[:,:,i], i+1, tol=1e-7)
		M_hat = np.linalg.multi_dot([U, U.T, Tensor[:,:,i], V, V.T])
		acc[i] = np.linalg.norm(Tensor[:,:,i] - M_hat) ** 2 / np.linalg.norm(Tensor[:,:,i]) ** 2
	time_record[rep] = time.time() - start_time
	mean_acc[rep] = acc.mean()





#####################################################################
###   DTA                                                         ###
###   DTA                                                         ###
###   DTA                                                         ###
#####################################################################

time_record = np.zeros(nrep)
mean_acc = np.zeros(nrep)

for rep in range(nrep):
	# holder for accuracy by T
	acc = np.zeros(T)

	# regenerate data
	for i in range(T):
		Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))
	
	# set starting point
	U, V = Us, Vs
	s1 = s2 = 0.1 * np.ones(r)

	start_time = time.time()
	for i in range(T):
		s1, U, s2, V = dta(s1, U, s2, V, Tensor[:,:,i])
		M_hat = np.linalg.multi_dot([U, U.T, Tensor[:,:,i], V, V.T])
		acc[i] = np.linalg.norm(Tensor[:,:,i] - M_hat) ** 2 / np.linalg.norm(Tensor[:,:,i]) ** 2
	time_record[rep] = time.time() - start_time
	mean_acc[rep] = acc.mean()


#####################################################################
###   GTRUTH                                                      ###
###   GTRUTH                                                      ###
###   GTRUTH                                                      ###
#####################################################################

time_record = np.zeros(nrep)
mean_acc = np.zeros(nrep)

for rep in range(nrep):
	# holder for accuracy by T
	acc = np.zeros(T)

	# regenerate data
	for i in range(T):
		Tensor[:,:,i] = M_mean + np.random.normal(scale = sigma, size = (m, m))

	U, V = Ut, Vt
	start_time = time.time()
	for i in range(T):
		M_hat = np.linalg.multi_dot([U, U.T, Tensor[:,:,i], V, V.T])
		acc[i] = np.linalg.norm(Tensor[:,:,i] - M_hat) ** 2 / np.linalg.norm(Tensor[:,:,i]) ** 2
	time_record[rep] = time.time() - start_time
	mean_acc[rep] = acc.mean()


