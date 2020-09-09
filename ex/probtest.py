# execfile("probtest.py"); 

from pylab import *
from prob import *
import scipy.stats as stats

# PARAMETERS

num = 5000
n = 200
lo = 0.0
hi = 1.0

# INPUT

x = linspace(lo, hi, n)
data = random( num )

PM = mass()
PM.setNormPrior(data, [n, n/2], 4)
PM.calcNormPost(data)
hx = PM.sample(3)

pcolor(PM.X[1], PM.X[0], PM.P)
xscale('log')
axis('tight')
show()
