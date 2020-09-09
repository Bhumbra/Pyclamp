# A Python script that parses TDF files to calculate sums from TDF MD entries

from iofunc import *
from lsfunc import *
from strfunc import *

# PARAMETERS

ipdn = '/home/admin/analyses/mnrc/vrrc/tdf/trains/'
ipex = '.tdf'
opdn = '/home/admin/analyses/mnrc/vrrc/tsv/int/trains/'
opex = '.tsv'

si = 2e-5

# INPUT

ipfn, ipst = lsdir(ipdn, ipex, 1)
n = len(ipfn)

for i in range(n):
  X = readDTFile(ipdn + ipfn[i])
  N2 = len(X)
  N = N2 / 2
  x = []
  for j in range(N):
    j2 = j * 2
    l, L = X[j2], listtable(X[j2+1])
    l1, l2 = l[-1], l[-2]
    ok = len(l2) > 5
    if ok: ok = l2[:3] == 'psa'
    if ok:
      li, ri = l2.find('['), l2.find(']')
      lr = l2[li+1:ri]
      lr = lr.replace(' ', '')
      ci = lr.find(',')
      lo, hi = lr[:ci], lr[ci+1:]
      coef = si * (float(hi) - float(lo))
      MD = L.retFlds('MD', dtype = float)
      SD = list(MD * coef)
      ci = strfind(l1, ',')
      if len(ci) == 2:
        ci = ci[1]
        lb = l1[ci+2:-1] if l1[ci+1] == ' ' else lb[ci+1:-1]
        lb = lb.replace("'", '')
        lb = lb.replace(")", '')
        SD = [lb] + SD
      x.append(SD[:])
    writeSVFile(opdn + ipst[i] + opex, x)

