from mplqt import *
from nnmod import *
from pgb import *

# PARAMETERS

M = np.matrix([[1., 4.], [2., 3.]])
n = 1000
N = 10
l = (0.005, 0.0005, 0.001)

ATF = 1

# INPUT

#S = np.matrix(np.random.normal(size = [2, n]))
S = np.matrix(np.random.random(size = [2, n])) - 0.5
X = M * S

# PROCESSING

self = imlayer(-1, ATF)
self.setData(X)
self.setLD(l)
self.setATF(ATF)
pb = cpgb()
W, w = self.learn(N, pgb=pb)

# OUTPUT

print((W,w))
figure()
subplot(2, 2, 1)
plot(S[0,:].T, S[1,:].T, '.')
subplot(2, 2, 2)
plot(X[0,:].T, X[1,:].T, '.')
subplot(2, 2, 3)
plot(self.A[0,:].T, self.A[1,:].T, '.')
subplot(2, 2, 4)
plot(self.D)



