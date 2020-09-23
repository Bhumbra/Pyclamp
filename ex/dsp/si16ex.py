from mplqt import *
from si16 import *

self = SI16('test.sp')
data = self.ReadIntData()
plot(data[0, -1, :])
