from qmod import *
from pgb import *
import time
#from pylab import *; #ion()
#import lbwgui as lbw

ipfn = '/home/admin/data/pairs/XLSX/100903p1Second.xlsx'
#ipfn = '/home/admin/141015n3_trimmed.tab'
nmax = 16
nres = 16

from pyqtgraph.Qt import QtGui, QtCore

#'''

app = QtGui.QApplication([])
self = qmodl()
_pgb = pgb()
a = self.readFile(ipfn, False)
#self.setRes(_nres = None, _ares = 1, _vres = 128, _sres = None, _rres = None, _qres = None)
self.setRes(nres, 9, 8, 128)
self.setLimits(None, None, None, [1, nmax])
self.setData(self.X, 11.)
self.setPriors(False, _pgb)
t0 = time.time()
self.calcPosts(_pgb, True)
t2 = time.time()
print("Time taken/s: " + str(t1-t0))

