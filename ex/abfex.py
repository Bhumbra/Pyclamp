import matplotlib as mpl
mpl.use("Qt4Agg")
import matplotlib.pyplot as mp
mp.ion()
import numpy as np

import abf
self = abf.ABF("test.abf")
data = self.ReadADCData()
chan = self.ReadChannelInfo()

x = np.arange(len(data[0][0]), dtype = float) * chan[0].samplint

mp.figure()
mp.subplot(2, 1, 1)
mp.plot(x, data[1].T)
mp.subplot(2, 1, 2)
mp.plot(x, data[0].T)
mp.show()

