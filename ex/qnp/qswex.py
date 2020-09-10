import pyplot
import pyqtplot as pq
from fpanal import *
import abf 
import inspect
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyanim

import numpy as np
#ion()

#dn = '/home/admin/results/rcmn/vs/abf/'
#fn = '130926n2_0001SpikeTrain1stim.abf'
dn = '/home/admin/data/pairs/ConditionalPP/'
fn = '110503p1_0001ic33.abf'
#dn = '/home/admin/data/'
#fn = '131213n2Ca2.abf'

ch = 0
roi = [0.005, 0.03]
roi = [0.035, 0.06]
#roi = [0.005, 0.025]
roispec = 1.4
fps = 25
nsw = 10
#yl = [-70., 40.]
yl = [-500., 50]
ABF = abf.ABF(dn+fn)
data = ABF.ReadIntData()
chan = ABF.ReadChannelInfo()
si = 2e-5

app = QtGui.QApplication([])
form = QtGui.QMainWindow()
gbox = pq.gbox()
form.setCentralWidget(gbox)

'''
onsets = np.arange(len(data[0]), dtype = int) * len(data[0][0]) + len(data[0][0])
Pw = pyplot.pywave()
Pw.setData(data[0], chan[0], onsets)
Pw.setPlot()
Pw.show()
'''

'''
dock0 = pq.dock("Graph")
gbox0 = dock0.addGbox()
area.add(dock0, 'bottom')
Pw = pyplot.pywave()
living = [[True] * len(data[0])]
select = [[False] * len(data[0])]
select[0][0] = True
select[0][-1] = True
embold = [[False] * len(data[0])]
Pw.setData(data[0], si, [], [living, select, embold])
marks = [np.arange(Pw.ne, dtype = int), array(np.round(linspace(0., Pw.ns-1, Pw.ne)), dtype = int)]
Pw.setPlot(parent=gbox0)
'''
'''
Pw = pyplot.pywave()
Pw.setData(data[0], chan[0])
g = Pw.setPlot(form=form, parent=gbox)
Pw.setOverlay(False)
Pw.setLims([[0., 0.015], [-200., 100.]])
Pw.setWave()
g.show()
'''

'''
Pw = pyplot.pywave2()
Pw.setData(data[ch], chan[ch])
Pw.setPlots(form=form, parent=gbox)
Pw.setPens(5)
Pw.setLims([[roi, yl], [roi, yl]])
for pw in Pw.pw:
  pw.setLabels(['x', 'y'], None, fontSize = 18, tickFontSize=18)
Pw.show()
'''

'''
Pw = pyplot.pywav()
Pw.setData([data[0], data[1]], [chan[0], chan[1]])
Pw.setPlots(form=form, parent=gbox)
Pw.setPens(5)
Pw.setLims([[roi, None], [roi, None]])
for pw in Pw.pw:
  pw.setLabels(['x', 'y'], None, fontSize = 18, tickFontSize=18)
Pw.show()
'''

#'''
Pw = pyplot.pywav2()
Pw.setData([data[0], data[1]], [chan[0], chan[1]])
Pw.setPlots(form=form, parent=gbox)
Pw.setPens(5)
Pw.setLims([[roi, None], [roi, None]])
for pw in Pw.pw:
  pw.setLabels(['x', 'y'], None, fontSize = 18, tickFontSize=18)
Pw.show()
#'''

#'''
self = pyanim.pywanim(Pw)
self.setROI(roi, roispec)
self.setFPS(fps)
self.calcInd(nsw)
self.animate()
self.saveAnim()
#'''

