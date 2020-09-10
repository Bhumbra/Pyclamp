from qmod import *
from iofunc import *
import time
ipfn = '/home/admin/data/pairs/XLSX/100903p1Second.xlsx'
X = readXLData(ipfn)
t0 = time.time()
self = qbay()
self.setRes([64, 16, 1])
self.setLimits([0.04, 0.96], [0.05, 1.], None, [1, 19])
self.setData(X, 10.)
self.setPriors([self.sres, self.vres, self.ares], None, None, False)
self.calcPosts()
dt = time.time() - t0
print ("Time taken (s): " + str(dt))

