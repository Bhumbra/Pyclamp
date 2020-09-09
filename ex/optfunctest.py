from optfunc import *
import matplotlib as mpl; mpl.use('qt4agg')
from pylab import *; ion()
import numpy as np

n = 1000
mint = 0.01
maxt = 3.0
nsd = 0.05
sx = 1.0
dx = 0.0

p = [-3., 0.02]


x = linspace(mint, maxt, n)
xs = x * sx + dx
y = exp0val(p, xs)
y += nsd*np.random.normal(size=len(y))
h = exp0fit(xs, y, 1)
hp = h[0]
hy = exp0val(hp, xs)

print(hp)


figure
plot(xs,y, 'b')
hold(True)
plot(xs,hy, 'r')



show()
