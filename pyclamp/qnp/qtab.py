from iofunc import *
from lsfunc import *
from fpfunc import *
import os
import numpy as np

# PARAMETERS

ipdir = '/home/admin/data/pairs/'
ipext = '.xlsx';
ipspecfile = 'spec.tab';
notOverWrite = True
opdirs = ['/home/admin/data/pairs/lop/','/home/admin/data/pairs/hip/']
opext = '.tab'

# INPUT

spec = readDTFile(ipdir + ipspecfile)[0]

# PROCESSING

n = len(spec)
fn = [[]] * n
lo = [[]] * n
hi = [[]] * n

for i in range(n):
  speci = spec[i]
  fn[i] = speci[0]
  lo[i] = numstr2list(speci[1])
  hi[i] = [] if len(speci) < 3 else numstr2list(speci[2])

# OUTPUT

for i in range(n):
  if notOverWrite and (os.path.isfile(opdirs[0] + fn[i] + opext) or os.path.isfile(opdirs[1] + fn[i] + opext)):
    pass
  else:
    data = readXLData(ipdir + fn[i] + ipext)
    lodata = [[]] * len(lo[i])
    for j in range(len(lo[i])):
      lodata[j] = nanravel(data[lo[i][j], :])
    hidata = [[]] * len(hi[i])
    for j in range(len(hi[i])):
      hidata[j] = nanravel(data[hi[i][j], :])
    if len(lo[i]):
      writeDTFile(opdirs[0] + fn[i] + opext, lodata)
    if len(hi[i]):
      writeDTFile(opdirs[1] + fn[i] + opext, hidata) 
