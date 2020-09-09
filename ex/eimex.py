from mplqt import *
from nnmod import *
from pgb import *

# PARAMETERS

M = np.matrix([[0.5, -2.], [1.5, -0.5]])
n = 1000
N = 25
l = (0.04, 0.004)

ATF = 2
EIK = -1

# INPUT

#S = np.matrix(np.random.normal(size = [2, n]))
#S = np.matrix(0.5 * np.random.laplace(size = [2, n]))
#S = np.matrix(np.vstack( [np.random.random(size = [1, n]) - 0.5, 0.5 * np.random.normal(size = [1, n])] ))
#S = np.matrix(np.random.random(size = [2, n])) - 0.5
#S = np.matrix(np.vstack( [np.random.random(size = [1, n]) - 0.5, 0.5 * np.random.laplace(size = [1, n])] ))

sx = np.hstack( [np.random.normal(size = [1, n/4])-2., np.random.normal(size = [1, 3*n/4])+2.] )
sy = np.hstack( [np.random.normal(size = [1, 3*n/4])-2., np.random.normal(size = [1, n/4])+2.] )
#sx = np.hstack( [np.random.normal(size = [1, n/4])-4., np.random.normal(size = [1, n/2]), np.random.normal(size = [1, n/4])+4.] )
#sy = np.hstack( [np.random.normal(size = [1, n/2])-2., np.random.normal(size = [1, n/4])+2, np.random.normal(size = [1, n/2])-2.] )
i = np.random.permutation(n)
S = np.matrix(np.vstack( [sx[:, i], sy[:, i]] ) )

X = M * S

# PROCESSING

self = eimlayer(1, ATF, EIK)
self.setData(X)
self.setLD(l)
self.setATF(ATF)
self.setEIK(EIK)
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



