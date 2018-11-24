from nnsup import *
from mplqt import *
from network import *
from time import time

X = np.random.rand(2, 1000) 
Y = np.random.rand(2, 1000) + np.vstack((0.7 * X[0, :] + 0.8* X[1, :], -0.5 * X[0, :] + -0.9* X[1, :]))
#Y = np.vstack((0.7 * X[0, :] + 0.8* X[1, :], -0.5 * X[0, :] + -0.9* X[1, :]))
minY = Y.min(axis = 1)
ranY = Y.max(axis = 1) - minY
Y = (Y - minY.reshape((2, 1))) / ranY.reshape((2, 1))

#self = sup0()
#self = sup1()
self = supebp([2])
#self.setA(ReLU, ReDU)
t0 = time()
self.supervise(X, Y, 10, 2, 1.)
self.supervise(X, Y, 10, 5, 0.5)
self.supervise(X, Y, 10, 10, 0.2)
self.supervise(X, Y, 20, 20, 0.1)
self.supervise(X, Y, 20, 50, 0.05)
self.supervise(X, Y, 20, 100, 0.02)
self.supervise(X, Y, 50, 200, 0.01)
self.supervise(X, Y, 50, 500, 0.005)
self.supervise(X, Y, 50, 500, 0.002)
'''
self.supervise(X, Y, 50, 500, 0.001, 0.5)
self.supervise(X, Y, 10, 2, 1., 0, 1.)
self.supervise(X, Y, 10, 5, 0.5, 0., 0.5)
self.supervise(X, Y, 10, 10, 0.2, 0, 0.25)
self.supervise(X, Y, 20, 20, 0.1)
self.supervise(X, Y, 20, 50, 0.05)
self.supervise(X, Y, 20, 100, 0.02)
self.supervise(X, Y, 50, 200, 0.01)
self.supervise(X, Y, 50, 500, 0.005, 0.1)
self.supervise(X, Y, 50, 500, 0.002, 0.2)
self.supervise(X, Y, 50, 500, 0.001, 0.5)
'''
print(''.join(str(time()-t0) + " s"))

A = self.forward(X)
if type(A) is tuple:
  Z, A = A

if ndim(A) == 3:
  s = A.shape
  A = A.T.reshape((s[1], s[0]))

Du = self.Du if type(self.Du) is not list else self.Du[0]
DW = self.DW if type(self.DW) is not list else self.DW[0]

subplot(2, 2, 1)
plot(Y[0,:], Y[1,:], 'r.')

subplot(2, 2, 2)
plot(A[0,:], A[1,:], 'r.')

subplot(2, 2, 3)
plot(DW)
yscale('log')

subplot(2, 2, 4)
plot(Du)
yscale('log')


