# A disynaptic simulator

import matplotlib as mpl
mpl.use('qt4agg')
from pylab import *; ion()
import numpy as np
from pyclamp.qnp.qsim import *;
from pyclamp.dsp.iofunc import *

N0 = 5 # number of inter-neurones
NS = 200 # number of sweeps
S = np.array([[1.0, 0.3], [0.8, 0.5], [0.5, 0.1]]) #P(release)
e0, n0, q0, u0, v0 = 1., 10., 1., 0.05, 0.05
e1, n1, q1, u1, v1 = 5., 4., 100., 0.10, 0.20
th = 5.

oppn = '/home/admin/q2sim.tab'

# INPUT

N = len(S)
S0 = S[:,0]
S1 = S[:,1]

Q0 = [[]] * N0
Q1 = [[]] * N0

lo = 1e-300

for i in range(N0):
  Q0[i] = binoqsim(e0, n0, q0, u0, v0, S0[0], [False, True])
  Q1[i] = binoqsim(lo, n1, q1, u1, v1, S1[0], [False, True])

R0 = np.empty((N0, N, NS))
R1 = np.empty((N0, N, NS))

for i in range(N0):
  R0[i] = Q0[i].simulate(NS, S0)
  R1[i] = Q1[i].simulate(NS, S1)

F0 = R0 < th
R1[F0] = 0.
r1 = np.sum(R1, axis = 0) + np.random.normal(0., e1, (N, NS))

if oppn is not None:
  if oppn:
    if type(oppn) is str:
      if len(oppn):
        writeDTFile(oppn, r1.T)

