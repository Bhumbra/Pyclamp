from optfunc import *
from wffunc import *
from pylab import *
import numpy as np


n = 1000
mn = -100.0
mx = 220.0
nsd = 1.0

p = array( [-0.0, 10.0, 200.0, -3.0])

x = linspace(mn, mx, n)
y = htanval(p, x)
y += nsd*np.random.normal(size=len(y))

fit = htanfit(x,y)
hp = fit[0]
print(hp)

hy = htanval(hp, x)

figure
plot(x, y, 'b')
hold(True)
plot(x, hy, 'r')
show()

