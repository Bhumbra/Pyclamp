from optfunc import *
from mplqt import *
import numpy as np


n = 1000
mn = 0.001
mx = 2.0
sd = 0.05

p = array( [1.0, 0.5, 2.0, 1.5])

x = linspace(mn, mx, n)
y = -exp2val(p, x)
y += sd*np.random.normal(size=len(y))

fit = exp2fit(x,y)
hp = fit[0]
print(hp)

hy = exp2val(hp, x)

figure
plot(x, y, 'b')
hold(True)
plot(x, hy, 'r')
show()

