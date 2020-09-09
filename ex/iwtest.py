#execfile("iwtest.py")
import inspect
import pyplot
import pyqtgraph as pg
import pyqtplot as pq
from fpfunc import *
from abf import *
from pyqtgraph.Qt import QtGui, QtCore

dn = '/home/admin/data/pairs/ConditionalPP/'
fn = '100908p1_0001ic33.abf'

mdf = ABF(dn+fn)
data = mdf.ReadIntData()
chan = mdf.ReadChannelInfo()
d = data[0][:,250:5000]


app = QtGui.QApplication([])
form = QtGui.QMainWindow()
gbox = pq.gbox()

'''
self = pyplot.pywave()
self.setData(data[0], chan[0])
self.setPlot(form=form)
''' 

self = pyplot.pywav()
self.setData(data, chan)
self.setPlots(form=form, parent=gbox)
self.setPens(self.pw[0].pens, 5)
self.show()


class scroll:
  dI = 10
  DI = 1000
  def __init__(self, _obj):
    self.I = -self.dI
    self.obj = _obj
  def func(self):
    i = self.I + self.dI
    self.I = i
    self.obj.pw[0].setLims([[self.I, self.I+self.DI], None])
    self.obj.pw[0].setWave()
    

A = pq.anim(self.gbox)
S = scroll(self)
A.setAnimFunc(S.func)

self.pw[0].setLims([[0, 1000], None])
self.pw[0].setLabels(["t/s", "I/pA"])
self.pw[1].setLabels([None, "V/mV"])
self.pw[0].setWave()

a = A.animate(250)
A.saveAnim()
 
