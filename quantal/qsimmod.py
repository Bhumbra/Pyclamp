import sys
sys.path.append('/mnt/ntfs/g/GSB/code/python/gen/')
from iofunc import *
import datetime
from qmod import *

# PARAMETERS

ipd = "/home/admin/data/Q/tdf/s/"
ipe = ".tdf"
opd = "/home/admin/analyses/Q/tab/s/"
ope = ".tab"

ares = 1
vres = 96
sres = 128
nres = 32
nmax = 92

e = 20.
ngeo = 1. # 0 = flat, 1 = reciprocal

# SCHEDULE

ipf, ips = lsdir(ipd, ipe, 1)
N = len(ipf)
ops = ips

for i in range(N):
  # INPUT
  X = readDTData(ipd+ipf[i])
  n = len(X)
  eQNqrg = np.empty((n, 6), dtype = float)
  for j in range(n):
    print(datetime.datetime.now().isoformat()+": " + str(i)+"/"+str(N) + ": " + str(j)+"/"+str(n))
    q = qmodl(X[j], e)
    q.setRes(nres, ares, vres, sres)
    q.setLimits(None, None, None, [1, nmax], ngeo)
    q.setPriors()
    q.calcPosts(True)
    _eQNqrg = [q.hate, q.parq, q.parn, q.hatq, q.hatr, q.hatg]
    eQNqrg[j, :] = np.copy(np.array([_eQNqrg], dtype = float))
    del q
  writeDTFile(opd+ops[i]+ope, [eQNqrg])

