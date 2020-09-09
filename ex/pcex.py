from pyclamp import pyclamp
from wffunc import *
from wcicanal import *
from optfunc import *
import gc
import time
import iwave as iw
import pylab as pl

N = 200000
dt = 0.000001
dr = "/home/admin/data/pairs/ConditionalPP/"



self = pyclamp()
self.setDataDir(dr)

self.setDataFiles(['100908p1_0003ic100.abf'])
self.setChannels([0, 1], [[2e-05, 0.305175766755, 0.0], [2e-05, 0.00305175766755, 0.0]])
self.readIntData()
self.setSelection([-194])
self.setClampConfig(0, 0)
self.psa(0, [496, 1052])

