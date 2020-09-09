from pyclamp import pyclamp
from wffunc import *
from wcicanal import *
from optfunc import *
from mplqt import *

#dr = "/ntfs/f/GSB/data/pairs/ConditionalPP/"
#dr = "/ntfs/f/GSB/data/mn_active"
#dr = "/ntfs/f/GSB/data/ECS/ABF/"
dr = "/ntfs/f/GSB/data/"

self = pyclamp()
self.setDataDir(dr)

self.setDataFiles(['140227n2_02ca1.abf'])
self.setChannels([0, 0], [[2e-05, 0.305175766755, 0.0], [2e-05, 0.305175766755, 0.0]])
self.readIntData()
self.setSelection([-135])
self.setClampConfig(0, 0)
self.baseline(0, [[14207, 14234], [15214, 15542]], 2)

