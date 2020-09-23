import pyqtplot as pq
import pyplot
from fpanal import *
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
y = np.random.rand(100)*10.

'''
app = QtGui.QApplication([])
self = pq.graph()
self.plot(x, y)
self.show()
'''


#'''
app = QtGui.QApplication([])
self = pyplot.pyscat(x, y)
self.setPlot()
self.setEllipse()
#'''

#s2 = self.scat(size=10, pen=pg.mkPen('w'), pxMode=True)
#s2.addPoints(x, y)

#self.cart.plot(x, y)
self.show()

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


