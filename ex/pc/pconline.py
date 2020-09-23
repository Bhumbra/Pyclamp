from pyclamp import pyclamp
from wffunc import *
from wcicanal import *
from optfunc import *
from listfunc import *
import matplotlib as mpl; mpl.use('qt4agg')
from pylab import *; ion()
from iplot import *

# PARAMETERS

dr = "/ntfs/f/GSB/data/mn_active"
fn = '140415n1_0005.abf'

# INPUT

self = pyclamp()
self.setDataDir(dr)

self.setDataFiles([fn])
self.setChannels([1, 0], [[5e-05, 0.00305175766755, 0.0], [5e-05, 0.000305175788071, 0.0]])
self.readIntData()
self.setSelection()
self.setClampConfig(0, 0)
self.trigger(0, 1, 0., 0.005) 
self.align(0, 0, [0, 10000000], True, 0.95)
self.triggers(0, [0, 10000000])

# PROCESSING

A = self.table;
E = A.retFlds('Episode', dtype = int)
T = A.retFlds('Times', dtype = float)
I = A.retFlds('StepCom', dtype = float)

# OUTPUT

figure()

m = E.max()
ns = m+1
nr = floor(sqrt(float(ns)))
nc = ceil(ns/nr)

nr, nc = int(nr), int(nc)
    
for k in range(ns):
  isubplot(nr, nc, k+1)
  K = nonzero(k == E)[0]
  if len(K) > 1:
    t = T[K]
    i = I[K]
    f = 1./diff(t)
    l = i[1:]
    print(i[0])
    n2 = argmax(l)
    plot(l[:n2], f[:n2], 'r')
    plot(l[n2:], f[n2:], 'b')
    xlabel(r'$\Delta I$/nA')
    ylabel(r'$f$/Hz')

