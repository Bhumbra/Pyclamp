import matplotlib as mpl
mpl.use('qt4agg')
from pylab import *; ion()
from fpfunc import *
import numpy as np
import scipy as sp
from time import time, sleep

# PARAMETERS

r = 100
n = np.unique(np.array(np.logspace(log10(2), log10(100000), 100), dtype = int))
N = 10000

# PROCESSING

m = len(n)
NT = 2
T = np.empty((NT, m), dtype = float)

for i in range(m):
  ni = n[i]
  X = ones(r, ni)
  t = time()
  for j in range(N):
    Y = np.fft.fft(X)
  T[0][i] = time() - t
  t = time()
  for j in range(N):
    Y = dft(X)
  T[1][i] = time() - t

ni = np.array(n, dtype = float)
B = T[0] / (ni * np.log(ni))
figure()
plot(ni, T[0])
hold(True)
xscale('log')
plot(ni, T[1])


