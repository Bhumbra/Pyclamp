import PyQt5
import pyclamp.gui.pyqtplot as pq
import pyqtgraph as pg
from pyclamp.dsp.fpanal import *
from time import *
import numpy as np

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
#'''
app = QtGui.QApplication([])
form = pq.BaseFormClass()
area = pq.area()
form.setCentralWidget(area)

dock0 = pq.dock("Graph")
gbox0 = dock0.addGbox()
self = pq.graph(parent=gbox0)
#self.setGUI(False)
area.add(dock0, 'bottom')

dock1 = pq.dock()
bbox1 = dock1.addBbox()
bbox1.addButton()
bbox1.setText(0, "Label")
area.add(dock1, 'bottom')

def done(ev):
  evc = ev.cursors[:]
  self.setCursor()
  #self.form.repaint()
  print("done")
  sleep(1.)   
  print(evc)

def clicked(ev):
  print("clicked")
  self.setCursor([2,2], done)


bbox1.Connect(0, clicked)
self.plot(x, y)
form.show()

#'''


#self.cart.plot(x, y)
#form.Widget.show()


#self0.show()
#self1.show()



