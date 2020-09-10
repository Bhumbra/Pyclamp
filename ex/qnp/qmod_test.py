import sys
import datetime
sys.path.append('/mnt/ntfs/g/GSB/code/python/gen/')
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()
from pyclamp.dsp.iofunc import *
from pyclamp.qnp.qmod import *

# PARAMETERS

ipd = "/tmp/bqa/data/Q/S/"
ipe = ".tab"
opd = "/tmp/bqa/analyses/Q/tdf/s/"
ope = ".tab"

ares = 1
vres = 128
sres = 128
nres = 16
nmax = 16

e = 20.
ngeo = 1. # 0 = flat, 1 = reciprocal

# SCHEDULE

ipf, ips = lsdir(ipd, ipe, 1)
N = len(ipf)
ops = ips
self = None

for i in range(N):
  # INPUT
  X = readDTData(ipd+ipf[i])
  n = len(X)
  eQNqrg = np.empty((n, 6), dtype = float)
  for j in range(n):
    print(datetime.datetime.now().isoformat()+": " + str(i)+"/"+str(N) + ": " + str(j)+"/"+str(n))
    x = X[j]
    self = qmodl(x, e)
    self.setRes(nres, ares, vres, sres)
    self.setLimits(None, None, None, [1, nmax], ngeo)
    self.setPriors()
    self.calcPosts(False)
    _eQNqrg = [self.hate, self.parq, self.parn, self.hatq, self.hatr, self.hatg]
    eQNqrg[j, :] = np.copy(np.array([_eQNqrg], dtype = float))
    break
  break

prob = [None] * 3
for i in range(3):
  prob[i] = self.PMRQ.P[i, :, ]

vals = {'q': self.q, 'r': self.r}
figure()

for i in range(3):
  subplot(2, 3, i+1)
  pcolor(
         np.ravel(vals['q']),
         np.ravel(vals['r']),
         prob[i][:-1, :-1], cmap=cm.jet,
        )
  colorbar()
  xlabel(r'$q$')
  ylabel(r'$n$')
  title(str(i))
  xscale('log')
  yscale('log')

subplot(2, 3, 5)
pcolor(
       np.ravel(vals['q']),
       np.ravel(vals['r']),
       self.PRQ.P[:-1, :-1], cmap=cm.jet,
      )
colorbar()
xlabel(r'$q$')
ylabel(r'$n$')
title(str(i))
xscale('log')
yscale('log')
