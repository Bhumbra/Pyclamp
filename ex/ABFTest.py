#execfile("ABFTest.py")

import matplotlib as mpl
import platform
#mpl.use('WXAgg')
mpl.use('Qt4Agg')
#mpl.rcParams['backend.qt4']='PySide'

from fpanal import *
import abf 
import iplot as ip
import iwave as iw
import pylab as pl
import matplotlib as mpl
import inspect
from pylab import *
#ion()
def callbackfunc(event = None):
  print("Button clicked")

dn = '/home/admin/data/pairs/ConditionalPP/'
if platform.platform().startswith("Windows"):
  dn = 'F:/GSB/data/pairs/ConditionalPP/'

fn = '100908p1_0001ic33.abf'

ABF = abf.ABF(dn+fn)
data = ABF.ReadADCData()
chan = ABF.ReadChannelInfo()
d = data[0][:100,550:800]
si = 2e-5
'''
dd = DDD(d, 2, [-1, 1])
rf = RFA(d, 2e-5, [-1, -1])
ft = FFT(d, [-1, 1])
pc = PCA(d, [-1, 1])

'''
'''
figure()
i = 0
i0 = round(rf.Z[i,10]/si)
i1 = round(rf.Z[i,15]/si)
y = rf.Y[i][i0:i1]
ny = len(y)
t = np.arange(ny)*si
p = rf.Z[i,26:31]
haty = exppulsefun(p, t)
plot(t, y)
hold('True')
plot(t, haty, 'r')

figure()
plot(d[i]);
#'''

#d[1] = data[1]

#d0 = data[1][0]

#wav = iw.iwav(d, chan[0])
#wav.subplot();

#wav = iw.iwavs(data[1], chan[1])
#wav.setPlots()

#wav = iw.iwavsel(d, chan[0])
#wav.subplot()

#wav = iw.iwavsel2(data[0], chan[0])
#wav.setPlots()

#wav = iw.imulwav( [ [data[0]], [data[1]] ], chan)
#wav.setLines('jet')
#wav.setSubPlots()
 
#wav = iw.imulwavsel2(data, chan, 0)
#wav.setFigs()

#wav = iw.iwavscat(d, chan[0], [d.min(axis = 1)], [d.max(axis = 1)])
#wav.setSubPlots()

#wav = iw.iwavscat(d, chan[0], [d.min(axis = 1), d.max(axis = 1)], [d.max(axis = 1), d.min(axis = 1)])
#wav.setSubPlots()

#wav = iw.iwavscatdisc(d, chan[0], [d.min(axis = 1)], [d.max(axis = 1)])
#wav.setSubPlots()
#wav.setEllipse()

#wav = iw.iwavscatdisc(d, chan[0], [d.min(axis = 1), d.max(axis = 1)], [d.max(axis = 1), d.min(axis = 1)])
#wav.setSubPlots()
#wav.setEllipse()

#wav = iw.iwavscatdiscbool(d, chan[0], [rf.Z[:,3], real(ft.Z[:,1]), pc.Z[:,0]], [rf.Z[:,22], imag(ft.Z[:,1]), pc.Z[:,1]])
#wav.setSubPlots()

wav = iw.iwavscatdiscboolcalc(d, chan[0])
wav.setSubPlots()

#na = ip.exes('t', 4, 5)
#but = mpl.widgets.Button(na, 'Button')
#but.on_clicked(callbackfunc)
pl.show()
