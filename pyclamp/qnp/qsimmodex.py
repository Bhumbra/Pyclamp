import sys
sys.path.append('/mnt/ntfs/g/GSB/code/python/gen/')
from iofunc import *
import datetime
from qmod import *
from mplqt import *

# PARAMETERS

ipd = "/home/admin/data/q/tdf/rnr/"
ipe = ".tdf"
opd = "/home/admin/analyses/Q/tab/rnr/r/"
ope = ".tab"

ares = 1
vres = 96
sres = 128
nres = 32
nmax = 92
sex = '0.2.'
sid = 3

e = 20.
ngeo = 0 # 0 = flat, 1 = reciprocal

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
    if ipf[i].find(sex) >= 0:
      if j == sid:
        q = qmodl(X[j], e)
        q.setRes(nres, ares, vres, sres)
        q.setLimits(None, None, None, [1, nmax], ngeo)
        q.setPriors()
        q.calcPosts(True)
        asd

