# -*- coding: utf-8 -*-
"""
Simple demonstration of TableWidget, which is an extension of QTableWidget
that automatically displays a variety of tabluar data formats.
"""
#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtplot import *

import numpy as np

app = QtGui.QApplication([])

form = BaseFormClass()
area = area()
form.setCentralWidget(area)

dock0 = dock("Dock0")
dock1 = dock("Dock1")
dock2 = dock("Dock2")
dock3 = dock("Dock3")
area.add(dock0, 'left')
area.add(dock1, 'right')
area.add(dock2, 'bottom', dock0)
area.add(dock3, 'bottom', dock1)
#self.setGUI(False)
w = tabl()
dock3.add(w)
#layout.addItem(w)
#form.setCentralWidget(area)
#dock.add(w)
form.show()
#w.resize(500,500)
#w.setWindowTitle('pyqtgraph example: TableWidget')


'''
data = np.array([
  (1,   1.6,   'x'),
  (3,   5.4,   'y'),
  (8,   12.5,  'z'),
  (443, 1e-12, 'w'),
  ], dtype=[('Column 1', int), ('Column 2', float), ('Column 3', object)])
'''  
data = np.random.random((100, 100))

w.setData(data)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  import sys
  if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()

