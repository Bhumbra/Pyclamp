# A Module to colate means from data formatted for quantal analysis

from iofunc import *
from qmod import *
from fpfunc import *
from lsfunc import *

# PARAMETERS

ipdn = '/home/admin/analyses/mnrc/vrrc/tdf/trains/'
ipex = '.tdf'
oppn = '/home/admin/results/mnrc/vrrc/tab_new/means.tab'

# INPUT

ipfn, ipfs = lsdir(ipdn, ipex, 1)

# PROCESSING

n = len(ipfn)

X = [[]] * (n*2)

for i in range(n):
  lbl, dat = readQfile(ipdn+ipfn[i])
  Dat = list2nan2D(dat) 
  mn = nanmean(Dat, axis = 1)
  X[i*2] = ['ID'] + lbl
  X[i*2+1] = [ipfs[i]] + list(mn)

# OUTPUT

writeDTFile(oppn, [X])

