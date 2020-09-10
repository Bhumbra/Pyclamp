import pyqtplot as pq
import pyqtgraph as pg
from fpanal import *
import abf 
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
'''
app = QtGui.QApplication([])
self = pq.graph()
self.plot(x, y)
self.show()
'''

'''
app = QtGui.QApplication([])
area = pq.mount()
#form.setCentralWidget(area)
w2 = area.addAxes()
s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
w2.addItem(s2)

s2.addPoints(x, y)

#plot = pq.graph()
#plot.plot(x, y)
area.show()
'''

app = QtGui.QApplication([])
self = pq.graph()
s2 = self.scat(size=10, pen=pg.mkPen('w'), pxMode=True)
s2.addPoints(x, y)

#self.cart.plot(x, y)
self.form.show()

'''
app = QtGui.QApplication([])
frm = QtGui.QMainWindow()
mnt = pg.GraphicsWindow(parent=frm)
self0 = pq.graph(parent=mnt)
s0 = self0.scat(size=10, pen=pg.mkPen('w'), pxMode=True)
s0.addPoints(x, y)
self1 = pq.graph(parent=mnt)
s1 = self1.scat(size=10, pen=pg.mkPen('w'), pxMode=True)
s1.addPoints(x, y)
'''

#self0.show()
#self1.show()


