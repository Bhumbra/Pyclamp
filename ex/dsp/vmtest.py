from fpfunc import *
from cstats import *
from pylab import *; ion()
from scipy.stats import *
from scipy.stats.morestats import *
import numpy as np

kran = [0.001, 1000.]
N = 10000
n = 200

k = geospace(kran[0], kran[1], n)
hk = empty(n, dtype = float)

for i in range(n):
  x = np.random.vonmises(0., k[i], N)
  hk[i] = ccon(x)

h = hk;

plot(k, h)
xscale('log')
yscale('log')

