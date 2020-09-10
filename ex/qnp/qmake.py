# A waveform maker from simuluated quantal data

from pyclamp.dnp.qsim import *
from pyclamp.dsp.tdf import *

# PARAMETERS

# Quantal

q = 100.
n = 6
u = 0.075
v = 0.
e0 = 0.001
S = [0.5]
e = 1.
N = 30

# Waveform

si = 2e-5
win = [-0.2, 0.8]
ldc = [7.5, 5.0]

# Write

oppn = "/home/admin/data/q/tdf/qmake.tdf"

# INPUT

Q = binoqsim(e, n, q, u, v, 0.5, [True, False])
X = Q.simulate(N, S)

# PROCESSING

W = amp2wave(X, si, e, win, ldc)

# OUTPUT

tdf = TDF()
_ = tdf.setData(W)
_ = tdf.writeData(oppn)

