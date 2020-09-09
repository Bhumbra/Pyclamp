# execfile("qmods.py")

# A multiple quantal analyser

import sys
sys.path.append("/home/admin/code/python/gen/") 
from pylab import *
from iofunc import *
from iplot import pbfig

import quantal
import os

# PARAMETERS

output = True
ipdirn = '/home/admin/data/Q/'
opdirn = '/home/admin/results/Q//'
sbdirn = ['e/', 'n/', 'u/', 'v/', 'a/', 'b/']

ipdirn = '/home/admin/data/Q/'
opdirn = '/home/admin/results/Q/'
sbdirn = ['m/']

extn = '.tab'

modelprob = 1
rescoef = 1.0
resmin = 128
resmax = 128
nmax = 16
edef = 25
nummax = 100
firstAndLastOnly = False
notOverWriteOutput = True

usefilter = False
filterList = ['S0.5u']
#filterList = ['N60m']

# INPUT

n = len(sbdirn)
IpFils = [[]] * n
N = [[]] *n

for i in range(n):
  IpFils[i] = os.listdir(ipdirn + sbdirn[i])
  for ipfil in IpFils[i]:
    if os.path.splitext(ipfil)[1] != extn:
      IpFils[i].remove(ipfil)  
  N[i] = len(IpFils[i])    

# PROCESSING AND OUTPUT

pb = pbfig("Analysing")
  
counter = 0;

for hn in range(n):
  ipfils = IpFils[hn]
  for iN in range(N[hn]):
    ipfil = ipfils[iN]
    if sbdirn[hn] == 'e/':
      ei = ipfil.find('e')
      ti = ipfil.find('t')
      e = float(ipfil[(ei+1):(ti-1)])
    else:
      e = edef
    opfil = ipfil
    datalist = readDTFile(ipdirn+sbdirn[hn]+ipfil)
    M = len(datalist)
    if usefilter:
      notanalyse = True
      for filterlist in filterList:
        if opfil.find(filterlist) >= 0:
          notanalyse = False
      if notanalyse:
        M = 0
    if not(nummax is None):
      if M > nummax:
        datalist = datalist[:nummax]
        M = len(datalist)
    if notOverWriteOutput and os.path.isfile(opdirn + sbdirn[hn] + opfil):
      M = 0     
    pb.setup(["Directory (" + sbdirn[hn] + ")", "File (" + ipfil + ")", "Repetition"], [n, N[hn], M], ['g', 'r', 'b'])  
    pb.update([hn, iN, 0])
    Q = [[]] * M
    if M:
      counter += 1
    for jM in range(M):
      pb.updatelast(jM)
      data = array(datalist[jM], dtype = float)
      if firstAndLastOnly:
        data = np.vstack([data[0], data[-1]])
      self = quantal.anal(data, e, nmax, modelprob, rescoef, resmin, resmax, False)
      Q[jM] = self.results
    if output and M > 0:
      writeDTFile(opdirn + sbdirn[hn] + opfil, Q)
pb.close()

print("Analysis complete.")
      
