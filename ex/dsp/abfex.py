import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as mp
mp.ion()
import numpy as np
from pyclamp.dsp.abf import ABF

self = ABF("/mnt/ntfs/f/GSB/data/GlyT2Adult/160603n2_0000.abf")
data = self.ReadADCData()
chan = self.ReadChannelInfo()

x = np.arange(len(data[0][0]), dtype = float) * chan[0].samplint

mp.figure()
for i in range(len(data)):
  mp.subplot(len(data), 1, i+1)
  mp.plot(x, data[i].T)
mp.show()

