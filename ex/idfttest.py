from pylab import *
from fpfunc import *
from optfunc import *

n = 1
dx = pi

m = n*2+1
m2 = n*4 + 1

P = 2.*arange(m) + 2.
p = np.empty(m2, dtype = float)
p[:m] = P
Z = tile(complex(P[0], 0.), m)
for i in range(n):
  Z[i+1] = complex(P[2*i+1], P[2*i+2])
  Z[-i-1] = conj(Z[i+1])
  p[-(2*i+2)] = p[2*i+1]
  p[-(2*i+1)] = -p[2*i+2]

x = linspace(-pi+dx, pi+dx, len(Z)+1)[:-1]
y = idftval(P, x)
hp = idftfit(x, y)

