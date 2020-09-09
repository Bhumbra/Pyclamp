import pyplot
import pyqtplot as pq
from fpanal import *
import abf 
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import numpy as np
#ion()

#dn = '/home/admin/results/rcmn/vs/abf/'
#fn = '130926n2_0001SpikeTrain1stim.abf'
dn = '/home/admin/data/pairs/ConditionalPP/'
fn = '100908p1_0001ic33.abf'
#dn = '/home/admin/data/'
#fn = '131213n2Ca2.abf'

ABF = abf.ABF(dn+fn)
data = ABF.ReadIntData()
chan = ABF.ReadChannelInfo()
si = 2e-5

app = QtGui.QApplication([])
form = QtGui.QMainWindow()
area = pq.area()
form.setCentralWidget(area)

'''
onsets = np.arange(len(data[0]), dtype = int) * len(data[0][0]) + len(data[0][0])
self = pyplot.pywave()
self.setData(data[0], chan[0], onsets)
self.setPlot()
self.show()
'''

'''
dock0 = pq.dock("Graph")
gbox0 = dock0.addGbox()
area.add(dock0, 'bottom')
self = pyplot.pywave()
living = [[True] * len(data[0])]
select = [[False] * len(data[0])]
select[0][0] = True
select[0][-1] = True
embold = [[False] * len(data[0])]
self.setData(data[0], si, [], [living, select, embold])
marks = [np.arange(self.ne, dtype = int), array(np.round(linspace(0., self.ns-1, self.ne)), dtype = int)]
self.setPlot(parent=gbox0)
'''
#'''
self = pyplot.pywave()
self.setData(data[0], chan[0])
g = self.setPlot(form=form)
g.show()
#'''

'''
self = pyplot.pywave2()
self.setData(data[0], chan[0])
'''

'''
self = pyplot.pywav()
self.setData([data[0], data[1]], [chan[0], chan[1]])
'''

'''
self = pyplot.pywav2()
self.setData([data[0], data[1]], [chan[0], chan[1]])
'''


def clicked0(ev = None):
  if isinstance(self, pyplot.pywave):
    self.setData(-data[0], si)
    self.setPlot()
  if isinstance(self, pyplot.pywav):
    self.setData([-data[0], -data[1]], si)
    self.setPlots()

def CB(ev = None):
  print(ev.cursors)
  self.setCursor()

def clicked1(ev = None):
  if isinstance(self, pyplot.pywave):
    self.setMarks(marks)
    self.setWave()
  if isinstance(self, pyplot.pywav):
    self.setCursor(0, [0, 0], CB)
    #self.setMarks([marks, marks])

'''
self.setPlots(parent=area)
dock1 = pq.dock()
bbox1 = dock1.addBbox()
bbox1.addButton()
bbox1.addButton()
bbox1.setText(0, "Label0")
bbox1.setText(1, "Label1")
area.add(dock1, 'bottom')
bbox1.Connect(0, clicked0)
bbox1.Connect(1, clicked1)
form.show()
'''


#'''
