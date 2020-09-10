from mplqt import *
from ses import *

#self = SES('010717n1.wcp')
self = SES('/home/admin/2015_04_21_0003.WCP')
data = self.ReadADCData()
plot(data[0, -1, :])

