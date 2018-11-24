from mplqt import *
from pylab import *
from fpanal import *

def dexp(x, dcp=0.5, dcn=1., c = 4., sd = 0.1): # dcp 0.5, dcn=1., c= 4. peaks at 1. with a 20-80 rise of 0.5386
  y = sd * np.random.normal(size = x.shape)
  d = np.zeros(x.shape, dtype = float)
  p = x >= 0. 
  d[p] += c * (exp(-dcp*x[p])-exp(-dcn*x[p]))
  return d + y 
  
t = [-5, 15]
si = 0.1
x = np.arange(t[0], t[1], si)
y = -dexp(x)

self = RFA()
self.analyse([y], mean(diff(x)))
i = argmax(y)
z = y[:i]
#print( si * (argmin(fabs(z-0.8))- argmin(fabs(z-0.2))) )


figure()
plot(x-t[0], y)
xlabel("IPI(Rise)="+str(self.Z[0,6])+"; IPI(Fall)="+str(self.Z[0,7])) # correct value 0.53859

