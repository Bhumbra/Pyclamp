from pylab import *
from wffunc import *

X = np.arange(100);
M = minmax2(float);
M.setData(X);
x_1, y_1 = M.retXY(0, inf, 2)
x0, y0 = M.retXY(0, inf, 5)
x1, y1 = M.retXY(0, inf, 10)
x2, y2 = M.retXY(0, inf, 20)
x3, y3 = M.retXY(0, inf, 40)
x4, y4 = M.retXY(0, inf, 80)
x5, y5 = M.retXY(0, inf, 160)

