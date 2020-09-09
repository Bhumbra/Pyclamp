from prob import *

'''
dims = (4, 2, 3)
N = np.prod(dims)
P = np.arange(N, dtype = float).reshape(dims)
P /= P.sum()
self = mass(P)
self.setPmat(True)
print(self.repa())
'''

dims = [6, 5, 4]
ndims = len(dims)
z = [np.arange(dims[i], dtype = float)-0.5*float(dims[i]-1) for i in range(ndims)]
pr = [zpdf(z[i]) for i in range(ndims)]
self = mass(True)
self.setX(z)
self.setp(pr)
M0 = self.marginalise(0)
M1 = self.marginalise(1)
M2 = self.marginalise(2)

