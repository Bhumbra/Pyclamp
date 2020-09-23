from display import DISPLAY
from bqa import *
from pgb import *
import time
#from pylab import *; #ion()
#import lbwgui as lbw

ipfn = '/home/admin/data/pairs/XLSX/100903p1Second.xlsx'
ipfn = '/home/admin/141007n1_trimmed.tab'
opfn = '/home/admin/141007n1_trimmed.pkl'
nmax = 80

from pyqtgraph.Qt import QtGui, QtCore

#'''
if DISPLAY: app = QtGui.QApplication([])
self = Qmodl()
self.readFile(ipfn, False)
self.setRes(80,  1, 16, 32, 32, 32)
#self.setRes(nmax, 1, _vres = 16, _sres = 64)
self.setLimits(None, None, None, [1, nmax])
self.setData(self.X, 11.)
#self.setData(self.X, 10.)
self.setPriors(False)
t0 = time.time()
self.calcPosts(pgb(True), False)
t1 = time.time()
print "Time take/s: " + str(t1-t0)
self.archive(opfn)
#plt = self.plotMoments()
#plt = self.plotHist()
#plt = self.plotMarg1D()
#plt = self.plotMarg2D()
#plt = self.plotHatValues()
#plt = self.showGUI()
#plt.show()
