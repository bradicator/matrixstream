from dta import dta
from incremental_td import incremental_TD
import numpy as np
import time

nrep = 10

# gen data from mech2
m, T, r = 100, 100, 10
sigma = 1
np.random.seed(5)
Ut, R = np.linalg.qr(np.random.normal(size = (m, 2*r)))
Vt, R = np.linalg.qr(np.random.normal(size = (m, 2*r)))
Ut1, Ut2 = Ut[:,:r], Ut[:,r:]
Vt1, Vt2 = Vt[:,:r], Vt[:,r:]


M_mean1 = Ut1.dot(np.diag([r-i for i in range(r)])).dot(Vt1.T)
M_mean2 = Ut2.dot(np.diag([r-i for i in range(r)])).dot(Vt2.T)

Tensor = np.zeros((m, m, T))
for i in range(T):
	Tensor[:,:,i] = (T-i)/T*M_mean1 + i/T*M_mean2 + \
			np.random.normal(scale = sigma, size = (m, m))

# prepare starting points for DTA and TD
# set r again here
r = 5
Ut1, Ut2 = Ut1[:,:r], Ut2[:,:r]
Vt1, Vt2 = Vt1[:,:r], Vt2[:,:r]
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
		Tensor[:,:,i] = (T-i)/T*M_mean1 + i/T*M_mean2 + \
			np.random.normal(scale = sigma, size = (m, m))
	
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
###   ITD Forget                                                  ###
###   ITD                                                         ###
###   ITD                                                         ###
#####################################################################
nrep = 1
time_record = np.zeros(nrep)
mean_acc = np.zeros(nrep)

for rep in range(nrep):
	# holder for accuracy by T
	acc = np.zeros(T)

	# regenerate data
	for i in range(T):
		Tensor[:,:,i] = (T-i)/T*M_mean1 + i/T*M_mean2 + \
			np.random.normal(scale = sigma, size = (m, m))
	
	# set starting point
	U, V = Us, Vs
	s1 = s2 = np.ones(r)

	start_time = time.time()
	for i in range(T):
		s1, U, s2, V = incremental_TD(s1, U, s2, V, Tensor[:,:,i], i+1, tol=1e-7, forget = 0.9)
		M_hat = np.linalg.multi_dot([U, U.T, Tensor[:,:,i], V, V.T])
		acc[i] = np.linalg.norm(Tensor[:,:,i] - M_hat) ** 2 / np.linalg.norm(Tensor[:,:,i]) ** 2
	time_record[rep] = time.time() - start_time;print(acc)
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
		Tensor[:,:,i] = (T-i)/T*M_mean1 + i/T*M_mean2 + \
			np.random.normal(scale = sigma, size = (m, m))
	
	start_time = time.time()
	for i in range(T):
		rho = i/T
		U = ((1-rho)*Ut1 + rho*Ut2) / np.sqrt(rho**2 + (1-rho)**2)
		V = ((1-rho)*Vt1 + rho*Vt2) / np.sqrt(rho**2 + (1-rho)**2)
		M_hat = np.linalg.multi_dot([U, U.T, Tensor[:,:,i], V, V.T])
		acc[i] = np.linalg.norm(Tensor[:,:,i] - M_hat) ** 2 / np.linalg.norm(Tensor[:,:,i]) ** 2
	time_record[rep] = time.time() - start_time
	mean_acc[rep] = acc.mean()


