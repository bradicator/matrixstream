def incremental_td2(U, U_egnval, V, V_egnval, x, n):
	f = 1 / n
	r1 = len(U_egnval)
	r2 = len(V_egnval)

	Z = x @ V
	ZZ = Z - U @ U.T @ Z
	Q, R = np.linalg.qr(ZZ)
	ZTU = Z.T @ U
	H = np.block([[(1-f)*np.diag(U_egnval) + f*ZTU.T @ ZTU, f*ZTU.T @ R.T],
		[f * R @ ZTU, f * R @ R.T]])
	egnval, Gamma = np.linalg.eigh(H)
	U_egnval_new = egnval[-r1:]
	U_new = np.hstack((U, Q)) @ Gamma[:,-r1:] 

	Y = x.T @ U
	YY = Y - V @ V.T @ Y
	Q, R = np.linalg.qr(YY)
	YTV = Y.T @ V
	H = np.block([[(1-f)*np.diag(V_egnval) + f*YTV.T @ YTV, f*YTV.T @ R.T],
		[f * R @ YTV, f * R @ R.T]])
	egnval, Gamma = np.linalg.eigh(H)
	V_egnval_new = egnval[-r2:]
	V_new = np.hstack((V, Q)) @ Gamma[:,-r2:]

	return U_new, U_egnval_new, V_new, V_egnval_new

#%% test case
np.random.seed(5)
a = np.random.normal(size=(10, 3))
egnval, egnvec = np.linalg.eigh(a @ a.T)
b = np.random.normal(size=(10, 3))
bb = b @ b.T
egnval2, egnvec2 = np.linalg.eigh(bb)
f1, f2, f3, f4 = egnvec, egnval, egnvec2, egnval2

#%%
x = bb + np.random.normal(scale = 0.1, size = (10, 10))
f1, f2, f3, f4 = incremental_td2(f1, f2, f3, f4, x, 200)

