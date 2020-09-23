from mplqt import *
from nnmod import *
from pgb import *

# PARAMETERS

M = np.matrix([[1., 4.], [2., 3.]])
n = 1000
N = 10

# INPUT

S = np.matrix(np.random.normal(size = [2, n]))
S = np.matrix(np.random.random(size = [2, n])) - 0.5
X = M * S

# PROCESSING

self = layer()
self.setData(X)
pb = cpgb()
W = self.learn(N, pgb=pb)

# OUTPUT

print(W)
figure()
subplot(2, 2, 1)
plot(S[0,:].T, S[1,:].T, '.')
subplot(2, 2, 2)
plot(X[0,:].T, X[1,:].T, '.')
subplot(2, 2, 3)
plot(self.A[0,:].T, self.A[1,:].T, '.')
subplot(2, 2, 4)
plot(self.D)



