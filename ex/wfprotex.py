import sys
sys.path.append('/home/admin/code/python/pyclamp/')
import matplotlib as mpl
mpl.use("Qt4Agg")
from pylab import *; ion()
from wfprot import *
from pyclamp import pyclamp


pc = pyclamp()
pc.setDataDir("/ntfs/f/GSB/data/Single_GlyT2")

#'''
pc.setDataFiles(['100209n3_0003.abf'])
pc.setChannels([0, 1], [[2e-05, 0.061035153351, 0.0], [2e-05, 0.0006103515625, 0.0]])
pc.readIntData()
pc.setSelection([-6])
pc.setClampConfig(0, 0)
wfp = protocol()
wfp.setSamp(pc.SiGnOf[1])
print(wfp.estProt(pc.Data[1]))
#'''

'''
ns = 1000
ne = 10
wfp = protocol()
wfp.initProt(ns, ne, 0)
wfp.addStep(ns/10, -400, ns/5, 10, 100)
wfp.addRamp(ns/10+ns/2, -500, ns/5, -300, 10, 100)
x = arange(ns)
X = tile(x.reshape((1, ns)), (ne, 1))
Y = copy(X)
for i in range(ne):
  Y[i] = wfp.rety(x, i)
figure()
plot(X.T, Y.T)
'''

