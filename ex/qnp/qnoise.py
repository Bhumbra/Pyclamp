# execfile("qnoise.py"); from pylab import *
# self.calcPosts(); 
# self.plotMargPosts(); self.plotCondPosts(); self.overlayHist(); pl.show();

#  Parabolic example

import numpy as np
import pylab as pl
import qsim
from qmod import *

#  PARAMATERS

i = 0;
e = pow(10., 2)
n1 = 5

nqvN0 = [5, -100., pow(30.0, 2), 100]
nqvN1 = [n1, 1.5*float(nqvN0[0]*nqvN0[1])/float(n1), pow(50.0, 2), 0]

slo = 0.05
shi = 0.75
sno = 5;

nx = 1000;

rescoef = 1.0;
resmax = 16

# INPUT

#qm = qsim.binoqsim(e, n, q, v)
qm0 = qsim.binoqsim(e, nqvN0[0], nqvN0[1], nqvN0[2])
qm1 = qsim.binoqsim(e, nqvN1[0], nqvN1[1], nqvN1[2])

# PROCESSING

self = qbay(rescoef, resmax)
#comp = qbay(rescoef, resmax)

s = np.linspace(slo, shi, sno)
QS0 = qm0.simulate(nqvN0[3], s)
QS1 = qm1.simulate(nqvN1[3], s)
QS = np.hstack( [QS0, QS1] )


Q = qrnbay()
Q.setData(QS)


#self.setData(QS, e)
self.setData(QS, e)
#comp.setData(QS, e)

mn = np.mean(QS, 1)
vr = np.var(QS, 1)

x = np.linspace(QS[i].min(), QS[i].max(), nx)
#y = ((nqvN0[3] + nqvN1[3]) * (x[1] - x[0]) * QLF(x, e, n, q, v, s[i]))


#self.calcPosts(); 
#comp.calcPosts(); 

# OUTPUT

#self.plotMargPosts(); self.plotCondPosts(); self.overlayHist();
#comp.plotMargPosts(); comp.plotCondPosts(); comp.overlayHist();

pl.figure()

pl.subplot(1, 2, 1)
pl.hist(QS[i], nx)
#pl.plot(x, y.ravel(), 'r.')

pl.subplot(1, 2, 2)
pl.plot(mn, vr, '.')
pl.show();


