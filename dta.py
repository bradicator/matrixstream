import numpy as np

def dtavec(U, s, x, f = 1):
	n = x.shape[1]
	r = U.shape[1]
	for i in range(n):
		xx = x[:, i]
		for j in range(r):
			y = np.dot(U[:, j], xx)
			s[j] = f * s[j] + y ** 2
			e = xx - y * U[:, j]
			U[:, j] += y / s[j] * e
			xx -= y * U[:, j]
			U[:, j] = U[:, j] / np.linalg.norm(U[:, j])
		U_new, R = np.linalg.qr(U)
		if not np.allclose(U_new[:,0], U[:, 0]):
			U_new[:,0] = -U_new[:,0]
	return U_new, s

def dta(s, U, s2, V, x, f = 1):
	U_new, s = dtavec(U, s, x, f)
	V_new, s2 = dtavec(V, s2, x.T, f)
	return s, U_new, s2, V_new