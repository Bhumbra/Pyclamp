# Module to simulate quantal data
from pyclamp.qnp.qsim import binoqsim
from pyclamp.dsp.iofunc import writeDTFile

outpath = '/tmp/bqa_test.tab'
e = 10.
n = 6
q = 100.
u = 0.1
v = 0.2
s = [0.2, 0.5, 0.8]
usegamma = [True, True]
num = 128

binom = binoqsim(e, n, q, u, v, 0.5, usegamma)
data = binom.simulate(num, s)
writeDTFile(outpath, [data])
