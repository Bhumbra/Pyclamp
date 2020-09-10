from mplqt import *
from smr import *

self = SMR('File_spike2_1.smr')
data = self.ReadADCData()
plot(data[0][:, :].T)

