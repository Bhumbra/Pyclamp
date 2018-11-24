from mplqt import *
from ses import *

self = SES('121207_001.EDR')
data = self.ReadADCData()
plot(np.ravel(data)[:1000])

