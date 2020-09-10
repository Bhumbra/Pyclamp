from numpy import *
import scipy.stats as stats
from mplqt import *
from fpfunc import *

N = 20
a = 20.
b = 0.5
mab = a / (a + b)
vab = a * b / (a + b)**2. /(a+b+1.)

X = [None] * N
Y = [None] * N
n = arange(N, dtype = int)
m = np.zeros(N, dtype = float)
v = np.zeros(N, dtype = float)


for i in range(1, N):
  #ni = float(n[i])
  #dn = 0.48/ni - 0.2/(ni**1.3) - 0.1/(ni**6.)
  #X[i] = np.linspace(dn, 1.-dn, n[i])
  X[i] = cumspace(n[i])
  Y[i] = stats.beta.ppf(X[i], a, b)
  m[i] = mean(Y[i])
  v[i] = var(Y[i])

figure()
subplot(2, 1, 1)
plot(n, m)
subplot(2, 1, 0)
plot(n, v)

