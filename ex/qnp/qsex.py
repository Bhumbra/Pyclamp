
import pyplot
import pyqtplot as pq
from fpanal import *
import tdf 
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyanim

import numpy as np
#ion()

# PARAMETERS

dn = '/home/admin/data/q/tdf/'
fn = 'qmake.tdf'
chan = [2e-5, 1., 1.]
ch = 0
roi = [0.18, 0.24]
roispec = 0.5
fps = 25
nsw = 30
#yl = [-70., 40.]
yl = [-50.,600]
FPS = 25

# INPUT

tdf = tdf.TDF(dn+fn)
_ = tdf.readData()
Data = tdf.readWave()
Chan = chan * len(Data)

# PROCESSING 

app = QtGui.QApplication([])
form = QtGui.QMainWindow()
gbox = pq.gbox()
form.setCentralWidget(gbox)

Pw = pyplot.pywave2()
Pw.setData(Data[ch], Chan[ch])
Pw.setPlots(form=form, parent=gbox)
Pw.setPens(5)
Pw.setLims([[roi, yl], [roi, yl]])
for pw in Pw.pw:
  pw.setLabels(['x', 'y'], None, fontSize = 18, tickFontSize=18)
Pw.show()
self = pyanim.pywanim(Pw)
self.setROI(roi, roispec)
self.setFPS(fps)
self.calcInd(nsw)
self.setRes(None, None, FPS)

# OUTPUT

self.animate()
self.saveAnim()

