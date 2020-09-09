
import pydisc
import abf 
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import pyqtplot as pq
import pyqtgraph as pg

import numpy as np
#ion()
def callbackfunc(event = None):
  print("Button clicked")

N = 2000

#dn = '/home/admin/results/rcmn/vs/abf/'
#fn = '130926n2_0001SpikeTrain1stim.abf'
dn = '/home/admin/data/pairs/ConditionalPP/'
fn = '100908p1_0001ic33.abf'

ABF = abf.ABF(dn+fn)
data = ABF.ReadADCData()
if N is not None: 
  N = min(N, len(data[0][0]))
  data = data[:,:,:N]
chan = ABF.ReadChannelInfo()
si = chan[0].samplint
ne = len(data[0])

x = np.min(data[0], axis = 1)
y = np.max(data[0], axis = 1)
true = np.ones(len(x), dtype = bool)
false = np.zeros(len(x), dtype = bool)

xy = np.empty((len(data[0]), 10))

for i in range(xy.shape[1]):
  xy[:,i] = np.random.random((len(data[0]))) * (i+2.)

app = QtGui.QApplication([])
form = QtGui.QMainWindow()
area = pq.area()
form.setCentralWidget(area)

#'''
self = pydisc.pywavescat()
self.setData([data[0]], si, [], [true, true, false], x, y)
self.setForm(form=form)
#'''
'''
self = pydisc.pyscat3()
self.setData([x, x, x], [y, y, y], [true, true, false])
self.setForm(form=form)
'''
'''
self = pydisc.pywavescat3()
self.setData([data[0]], si, [], [true, true, false], [x, x, x], [y, y, y])
self.setForm(form=form, parent=area)
'''

'''
self = pydisc.pydisc1()
self.setData(data[0][:, 600:800], si, [], [true, true, false])
self.setAnal(si = si)
self.setForm(form=form)#, parent=area)
'''
'''
self = pydisc.pydisc3()
self.setData(data[0][:, 600:800], si, [], [true, true, false])
self.setAnal(si = si)
self.setForm(form=form)#, parent=area)
'''

form.show()

#pg.plot(np.tile(data0, 10), pen=(0,3))  ## setting pen=(i,3) automaticaly creates three different-colored pens

