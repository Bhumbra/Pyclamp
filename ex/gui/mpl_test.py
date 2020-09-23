import matplotlib as mpl
mpl.use("Qt4Agg")
from pylab import *; ion()

pc = 2
dx = pc

x = linspace(-dx, dx, 100000)
y = tanh(x)
plot(x, y, 'b')
hold(True)
plot([x[0], x[-1]], [y[0], y[-1]], 'r')

