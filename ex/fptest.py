#execfile("fptest.py")

md = None
from fpfunc import *

import matplotlib as mpl
#mpl.use('WXAgg')

from iofunc import *

import MDABF as md
import iplot as ip
import iwave as iw
import pylab as pl
import matplotlib as mpl
import inspect
from pylab import *

def callbackfunc(event = None):
  print "Button clicked"

dn = '/home/admin/Data/ECS/ABF/'
fn = '071022n4_01Spike1.abf'
of = 'test.bin'


mdf = md.ABF(dn+fn)
data = mdf.ReadIntData()
chan = mdf.ReadChannelInfo()
d = data[0][:,250:]
#writeBinFile(dn+of, d)

D = copy(d)

#writeBinFile(dn+of, d)
pca = PCA()
pca.setCentreScale((0, 0))
pca.analyse(D)
#d[1] = data[1]

plot(pca.Z[:,0], pca.Z[:,1], '.')
