import sys
sys.path.append('/home/admin/code/python/pyclamp/')
import matplotlib as mpl
mpl.use("Qt4Agg")
from pylab import *; ion()
from wfprot import *
from pyclamp import pyclamp
from wcicanal import *


pc = pyclamp()
pc.setDataDir("/ntfs/f/GSB/data/Single_GlyT2")

pc.setDataFiles(['100311n5_0010.abf'])
pc.setChannels([0, 1], [[2e-05, 0.0305175772155, 0.0], [2e-05, 0.000305175788071, 0.0]])
pc.readIntData()
pc.setSelection([-168])
pc.setClampConfig(0, 0)
pc.trigger(0, 1, -5.63310681292, 0)
pc.align(0, 0, [2287, 5540])
pc.psa(0, [2677, 2838])
pc.writetable('100311n5.tab', '/ntfs/f/GSB/data')

'''
data = pc.extract(0, [3649, 7459])
ss = spikeShape(2e-5)
ss.analyse(data)
ss.sumPlot()
'''
