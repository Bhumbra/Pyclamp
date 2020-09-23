from pylab import *
from prob import *
x = linspace(-1, 10, 1000)
y = gampdf(x, 5.0, 1.0)
plot(x,y); show();
