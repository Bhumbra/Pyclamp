# Quantal analysis wrapper function

import iplot
import numpy as np
import matplotlib.pyplot as mp
import qmod 
import os
from fpfunc import *
from iofunc import *
from lsfunc import *
  
def anal(data, e = None, nmax = None, modelprob = 2, rescoef = 1.0, resmin = 128, resmax = 192, showfig = True, numformat = "%.3f"):
  modelbeta = modelprob == 2
  self = qmod.qbay(rescoef, resmin, resmax)
  self.setData(data, e, nmax)
  hata = np.NaN
  if modelprob:
    self.setPriors(rescoef, resmin, resmax, modelbeta)
    self.calcPosts()
    if self.modelBeta:      
      hata = self.hata 
  self.results = [] 
  self.results.append([r'$N$', str(int(self.nx.min())) + "-" + str(int(self.nx.max()))])
  self.results.append([r'$\vert \mu \vert$', str(int(self.NX))])    
  self.results.append([r'$<\epsilon>$', numformat % float(self.hate)])
  self.results.append([r'$<Q>$', numformat % float(self.parq)])
  self.results.append([r'$<U>$', numformat % float(self.paru)])
  self.results.append([r'$<N>$', numformat % float(self.parn)])
  self.results.append([r'$<V>$', numformat % float(self.parcv)])
  self.results.append([r'$<Q_v>$', numformat % float(self.parcq)])
  self.results.append([r'$<U_v>$', numformat % float(self.parcu)])
  self.results.append([r'$<N_v>$', numformat % float(self.parcn)])
  self.results.append(['Resn', str(int(self.res))])
  self.results.append([r'$\hat{q}$', numformat % float(self.hatq)])
  self.results.append([r'$\hat{\gamma}$', numformat % float(self.hatg)])
  self.results.append([r'$\hat{\alpha}$', numformat % float(hata)])
  self.results.append([r'$\hat{r}$', numformat % float(self.hatr)])
  self.results.append([r'$\hat{v}$', numformat % float(self.hatv)])
  self.results.append([r'$\hat{\lambda}$', numformat % float(self.hatl)])  
  self.results.append([r'$\hat{n}$', numformat % float(self.hatn)])
  if not(showfig): return self
  self.plotSummary()
  #'''
  return self         

def analfile(fn, e = None, nmax = None, modelprob = 0, rescoef = 1.0, resmin = None, resmax = None, showfig = True, numformat = "%.2f"):
  ext = os.path.splitext(fn)[1].lower()
  if ext == '.xls' or ext == '.xlsx':
    data = readXLData(fn)
  else:
    _data = readDTData(fn)
    if nDim(_data) < 3:
      data = _data
    else: # remove redundant singleton dimensions
      if len(_data) == 1:
        data = _data[0]
      else:
        maxn = 0
        for i in range(len(_data)):
          maxn = max(maxn, len(_data[i]))
        if maxn != 1:
          data = _data
        else:
          data = [[]] * len(_data)
          for i in range(len(_data)):
            data[i] = _data[i][0]
  if resmin == None: 
    resmin = 256/len(data)
  if resmax == None:  
    resmax = resmin * 2
  return anal(data, e, nmax, modelprob, rescoef, resmin, resmax, showfig, numformat)

