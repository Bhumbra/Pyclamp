# A BQA result tabulator

from qmod import *
from iofunc import *
from pgb import cpgb
import numpy as np

# PARAMETERS

ipdn = '/mnt/share/BQA/results/'
oppn = '/mnt/share/BQA/gr/bqa.tab'

ipex = '.pkl'
lbl = ['ID', 'e', 'q', 'g', 'r', 'n', 'v', 'S', 's']

# INPUT

ipfn, ipst = lsdir(ipdn, ipex, 1)
n = len(ipfn)

# PROCESSING

x = [[]] * n

pgb = cpgb("Processing PKL files", n)

for i in range(n):
  pgb.set(i)
  Id, fn = ipst[i], ipfn[i]
  self = qmodl()
  self.archive(ipdn+fn)
  ss = np.minimum(np.sort(self.hats), 1.)
  Ss = [ss[0], ss[-1]] if len(ss) > 1 else [ss[0], np.nan]
  x[i] = [Id, self.hate, self.hatq, self.hatg, self.hatr, self.hatn, self.hatv, Ss[0], Ss[1]]
  del self

pgb.close()

# OUTPUT

X = [lbl] + x
writeDTFile(oppn, [X])

