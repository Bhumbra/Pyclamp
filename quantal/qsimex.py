# execfile("qsimex.py");

#  Quantal simulation example

from pylab import *
from iofunc import *
from qmod import *

import matplotlib.pyplot as mp
import numpy as np
import qsim
import quantal
import os
import scipy.stats as stats

#  PARAMETERS

n = 6
q = 100

u = 0.3
v = 0.0
e = 10.0

num = 30
sno = 3;
slo = 0.10
shi = 0.45

usegammasim = [False, True] # [Intra-, Inter-]
usebeta = False
a = 0.5;

modelprob = 1
rescoef = 1.0;
resmin = 128
resmax = 192
nmax = 10

pf = '/home/admin/Test/test.tab'
fno = '/home/admin/Test/out.tab'

# INPUT

s = np.linspace(slo, shi, sno)
#s = np.array( [0.4, 0.5, 0.6, 0.45]);
#sno = len(s)

if not(usebeta):
  qm = qsim.binoqsim(e, n, q, u, v, 0.5, usegammasim)  
else:
  qm = qsim.betaqsim(e, n, q, u, v, a, 0.5, usegammasim)
  
Q = np.empty(n, dtype = float)
for i in range(n):
  Q[i] = qm.QUS[i].q

# PROCESSING
mino = 1e-300
l = v / (q+mino)
g = q / (l+mino)

QS = qm.simulate(num, s)
writeDTFile(pf, QS)

self = quantal.analfile(pf, e, nmax, modelprob, rescoef, resmin, resmax)

# OUTPUT

if modelprob: self.plotCondHist()
 