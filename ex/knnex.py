# K-nearest neighbours tester

from mplqt import *
from fpanal import *


# PARAMETERS

n = [100, 100]
m = [[5, 7], [7, 5]]
s = [[1, 1], [1, 1]]
k = 10
vr = 2
#vr = 2
#k = 10
vr = 1./3.


si = ['y.', 'c.']
so = ['ro', 'bo', 'mo']
ms = 16


# INPUT

N = len(n)
R = [[]] * N

for j in range(N):
  R[j] = [[]] * len(m[j])
  for i in range(len(m[j])):
    R[j][i] = np.random.normal(m[j][i], s[j][i], n[j])
  R[j] = np.matrix(R[j]).T

# PROCESSING

self = KNN()
self.setKVR(k, vr)
b = self.setRef(R)
ref = self.Ref

# OUTPUT

figure()
for i in range(N+1):
  if i == N:
    o = argtrue(np.product(b, axis=1))
    print(len(o))
  else:
    j = 0 if i else 1
    o = argtrue(np.logical_and(b[:,i], np.logical_not(b[:,j])))
  if not(i): hold(True)
  plot(ref[o,0], ref[o,1], so[i], markersize = ms)
for i in range(N):
  plot(R[i][:,0], R[i][:,1], si[i], markersize = ms)

