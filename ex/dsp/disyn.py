from prob import *
import matplotlib as mpl; mpl.use('qt4agg')
from pylab import *; ion()

# PARAMETERS

P0 = 0.8
N0 = 8

P1 = 0.1
N1 = 8

# INPUT

S = mass(tuple([P0]*N0)) # not
s = S.calcCMF()          # used

o = mass(P0)
R = mass(tuple([P1]*N1))
r = R.calcCMF()
c = o.condition(r)

# PROCESSING

mx = c.X * N0
mp = [c.P] * N0
P = mass()
P.setX(mx)
P.setp(mp)

p = P.calcCMF() 
hp = stats.binom.pmf(p.X[0], N0*N1, P0*P1)

# OUTPUT

#'''
figure()
plot(p.X[0], p.P, 'b')
hold(True)
plot(p.X[0], hp, 'r')
#'''

