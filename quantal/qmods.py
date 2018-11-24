# A function to run batch-run qmod runs

import qmod
import os
import pgb

from iofunc import *
from strfunc import *

DEFIPD = '/mnt/share/BQA/analyses/'

def qmods(_e = 1., _sres = 128, _vres = 64, _ares = 1, _nres = 32, _nmax = None, _ngeo = None, noprompt = 0, ipd = None, opd = None):

  if ipd is None: ipd = DEFIPD
  if ipd[-1] != '/': ipd += '/'
  if opd is None:
    if ipd.find('analyses') < 0: raise ValueError("Unknown output directory: " + ipd)
    if not(os.path.exists(ipd)): raise ValueError("Non-existent output directory: " + opd)
    opd = ipd.replace('analyses', 'results')
    if not(os.path.exists(opd)): raise ValueError("Non-existent output directory: " + opd)
  if _nmax is None: _nmax = _nres
  if _ngeo is None: _ngeo = 1

  print("Settings:")
  print("  Baseline noise standard deviation: " + str(_e))
  print("  Resolution for probability of release: " + str(_sres))
  print("  Resolution for coefficent of variation: " + str(_vres))
  print("  Resolution for heterogenous model alpha: " + str(_ares))
  print("  Resolution for number of release sites: " + str(_nres))
  print("  Maximum number of release sites: " + str(_nmax))
  print("  Reciprocal prior for number of release sites: " + str(_ngeo))
  print(' ')
  print("  Input directory: " + ipd)
  print("  Output directory: " + opd)
  print(' ')
  if _nres > _sres:
    print("Warning: a release site resolution exceeds that for the probability of release")
    print(' ')

  _ipfn, _ipst, _ipen = lsdir(ipd, ['.tab', '.tdf', '.tsv', '.xls', '.xlsx'], 2)
  _opfn = [st+".pkl" for st in _ipst]
  _n = len(_ipfn)
  for i in range(_n):
    print(str(i) + ": " + _ipfn[i])
  if not(_n):
    print("Input directory contains no identifiable data files.")
    return
  for fn in _opfn:
    havewarning=False
    if os.path.isfile(opd+fn):
      print("Warning: existing output file detected: " + fn)
  _ui = '' if noprompt else input("Select file number(s) or hit `Enter' for batch analysis (Ctrl-D to exit): ")
  ui = uintparseunique(_ui, _n)
  n = len(ui)
  ipfn, opfn = [[]] * n, [[]] * n
  for i in range(n):
    ipfn[i], opfn[i] = _ipfn[ui[i]], _opfn[ui[i]]
  for i in range(n):
    ifn, ofn = ipfn[i], opfn[i]
    print(' ')
    print("Analysing " + ifn + " (" + str(i+1) + "/" + str(n) + ")")
    self = qmod.qmodl()
    self.readFile(ipd+ifn, False)
    self.setRes(_nres, _ares, _vres, _sres)
    self.setLimits(None, None, None, [1, _nmax], _ngeo)
    self.setData(self.X, _e)
    _pgb = pgb.pgb()
    self.setPriors(False, _pgb)
    self.calcPosts(None, _pgb)
    _pgb.close()
    print("Writing results to: " + ofn)
    self.archive(opd+ofn)
    del self
  print(' ')
  print("Analyses run complete.")

