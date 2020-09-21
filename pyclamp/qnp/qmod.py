import os
import sys
import webbrowser
import time
import numpy as np
import scipy as sp
import scipy.stats as stats
import numpy as np
import scipy as sp
import scipy.stats as stats

from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.iofunc import *
from pyclamp.qnp.bqa_dgei import BQA
import pyclamp.dsp.discprob as discprob
from PyQt4 import QtGui
from PyQt4 import QtCore
from pyclamp.gui.pgb import pgb
import pyclamp.gui.lbwgui as lbw
import pyqtgraph as pg
import pyqtgraph.opengl as gl # ImportError means python-opengl is missing
import pyclamp.gui.pyqtplot as pq

CWDIR = os.getcwd()
MANDIR = CWDIR + '/../pyclamp/man/'
MANPREF = "pyclamp"
MANSUFF = ".html"
MANHELP = ["su12", "su13", "su14", "su15", "su16", "su17"]

USEPARALLEL = False # default None

try:
  from joblib import Parallel, delayed
  MAXTHREADS=16
except ImportError:
  Parallel = None
  delayed = None
  MAXTHREADS=1

def parafun(x, a, b = 0., c = 0.):
  return a*x**2.+b*x+c

def parafit(x, y, ab = None, s = None):
  if ab is None:
    a = -1.
    b = nanmean(x)
  else:
    a, b = ab
  if s is None:
    return sp.optimize.curve_fit(parafun, x, y, p0 = (a,b))
  else:
    return sp.optimize.curve_fit(parafun, x, y, p0 = (a,b), sigma = s)

def QLF(_x, _e, _n, _g, _l, _a, _s):
  amax = 1e5
  x, e, n, g, l, a, s = np.array(_x, dtype = float), float(_e), int(round(_n)), float(_g), float(_l), float(_a), float(_s)
  pol = np.sign(l)
  x *= pol
  l *= pol
  s = max(0.0, s)
  s = min(1.0, s)
  if _a is None: _a = 2.*amax
  if np.isnan(_a): _a = 2.*amax
  if _a > amax:
    c = stats.binom.pmf(np.tile(np.arange(n+1).reshape(n+1, 1), (1, len(x))), n, s)
  else:
    b = a / s - a
    c = discprob.betanompmf(n, a, b)
  p = np.zeros(x.shape, dtype = float)
  for i in range(n+1):
    if not(i):
      p += c[i] * stats.norm.pdf(x, 0.0, e)
    else:
      p += c[i] * discprob.gampdf(x, g*float(i), l)
  return p

def QLV(_m1, _e, _n, _g, _l, _a = None):
  onem = 1e-16
  mino = 1e-300
  amax = 1e5
  bmin = 1e-16
  bmax = 1e16
  if _a is None: _a = amax*2.
  if np.isnan(_a): _a = amax*2.
  m1, e, n, g, l, a = nanravel(_m1), float(_e), int(round(_n)), float(_g), float(_l), float(_a)
  M = len(m1)
  q = g * l
  r = float(n) * q
  s = m1 / r
  s[s < onem] = onem
  s[s > (1.-onem)] = 1.-onem
  if a > amax:
    I = np.tile(np.arange(n+1).reshape(1, n+1), (M, 1))
    N = np.tile(n, (M, n+1))
    S = np.tile(s.reshape(M, 1), (1, n+1))
    c = stats.binom.pmf(I, N, S)
  else:
    b = a / (s+mino) - a
    b[b < bmin] = bmin
    b[b > bmax] = bmax
    a = np.tile(a, (M))
    c = discprob.betanompmf(n, a, b)
  x = np.tile(np.linspace(0.0, r, n+1).reshape(1, n+1), (M, 1))
  cx = np.sum(c*x, axis = 1)
  m2 = np.sum(c*(x-np.tile(cx.reshape(M, 1), (1, n+1)))**2.0, axis = 1)
  for i in range(n+1):
    if not(i):
      m2 += c[:,i] * e ** 2.0
    else:
      m2 += c[:,i] * g * float(i) * l ** 2.0
  return m2


def qlfun(p, x, i, e, n, opts): # opts = 0 returns value, 1 returns derivative, 2 returns both
  # p = [g, l, s_1, s_2, etc...]
  g = p[0]
  l = p[1]
  s = p[2:]
  si = s[i]
  C = np.empty( (n, len(x)), dtype = float)
  for j in range(n):
    C[j] = stats.binom.pmf(j, n, si)
  y = C[0] * stats.norm.pdf(x, 0., e)
  for j in range(1, n):
    y += C[j] * discprob.gampdf(x, g*float(i), l)
  if not(opts):
    return y

def qlval(p, x, i, e, n, y = None): # returns residual if y is entered
  maxo = 1e300
  if y == None: return qlfun(p, x, i, e, n)
  if p[0] <= 0.: return maxo
  if p[1] <= 0.: return maxo
  if min(p[2:]) <= 0.: return maxo
  if max(p[2:]) >= 1.: return maxo
  return qlfun(p, x, i, e, n) - y

def ParExpSum(i, x, Lgkj, Lrkj, Gdj, z = 0.):
  xi = x[i]
  if xi <= 0: return z
  lxi = np.log(xi)
  #Gm = discprob.gampdf(x, self.Gi[i], self.Ls[k][i])
  #y = np.exp(- g * np.log(l) - x/l - sp.special.gammaln(g) + (g-1.0)*np.log(x) )
  return np.exp(Lgkj + Lrkj*xi + Gdj*lxi)

def readQfile(pf = "", spec = None): # an attempt of a universal import function for quantal data
  #spec=None (or False) specifies row-arranged TAB data
  _Labels = None
  if not(len(pf)): return
  ipdn, ipfn = os.path.split(pf)
  ipfs, ipfe = os.path.splitext(ipfn)
  ipfe = ipfe.lower()
  _path = ipdn
  _stem = ipfs
  # Handle easiest (or rather least-flexible) case - an Excel file
  if ipfe == ".xls" or ipfe == ".xlsx":
    _Data, _Labels = readXLData(pf, 0, True) # this automatically tranposes and replaces strings with NaNs
    if not(np.all(np.isnan(_Data[:, 0]))): # if any columns are numeric do not assume headers as labels
      _Labels = [''] * len(_Data)
      for i in range(len(_Data)):
        _Labels[i] = '#' + str(i)
    return _Labels, _Data
  if ipfe == ".tsv": # tab separated values
    if type(spec) is bool: spec = None # ignore tab-orientation flags
    if spec is None: spec = 0          # use first 2D table if unspecified
    _Data = readSVFile(pf)[spec]
    _Labels = [''] * len(_Data)
    heads = True
    for _data in _Data:
      if type(_data) is not list: heads = False
    if heads:
      for i in range(len(_Data)):
        _Labels[i] = _Data[i][0]
        _Data[i] = _Data[i][1:]
    else:
      for i in range(len(_Data)):
        _Labels[i] = '#' + str(i)
    return _Labels, _Data
  # Otherwise it's a tab-delimited file
  _Data = readDTFile(pf)
  # Check if result file
  rsltfile = True
  lend = len(_Data)
  lend2 = lend//2
  if ipfe == ".tdf" or ipfe == ".tab": # *.tab delimited file extenstion
    if lend > 0: # has multiple pages
      if lend % 2 == 0: # of even number
        for i in range(lend2): # check for tuple/list alternation
          i2 = i*2
          if type(_Data[i2]) is not tuple: rsltfile = False
          if type(_Data[i2+1]) is not list: rsltfile = False
      else:
        rsltfile = False
    else:
      rsltfile = False
  else:
    rsltfile = False
  if not(rsltfile):
    _Data = readDTData(_Data)
    while nDim(_Data) > 2 and len(_Data) == 1:
      _Data = _Data[0]
    _Labels = [''] * len(_Data)
    heads = True
    for _data in _Data:
      if type(_data) is not list: heads = False
    if heads:
      for i in range(len(_Data)):
        _Labels[i] = _Data[i][0]
        _Data[i] = _Data[i][1:]
    else:
      for i in range(len(_Data)):
        _Labels[i] = '#' + str(i)
    return _Labels, _Data
  # Read result file
  if type(spec) is bool: spec = 'MWD'
  if spec is None: spec = 'MWD'
  s_Data = [[]] * lend2
  _Labels = [''] * lend2
  for i in range(lend2):
    _log = _Data[i*2]
    k = -1
    l = -1
    for j in range(len(_log)):
      if _log[j].find('setLabel(') >= 0:
        k = j
      elif _log[j].find('writetable(') >= 0:
        if len(strfind(_log[j], ',')) == 2:
          l = j
    if k < 0 and l < 0:
      _Labels[i] = str(i)
    else:
      if k > l:
        _label = _log[k]
        _label = _label.replace("setLabel('", "")
        _label = _label.replace("')", "")
      else:
        _label = _log[l]
        iq = strfind(_label, "'")
        _label = _label[iq[-2]+1:iq[-1]]
      _Labels[i] = str(i) + " (" + _label + ")"
    _data = _Data[i*2+1]
    if len(_data): k = listfind(_data[0], spec)
    if k is None:
      raise ValueError("File " + ipfn + " missing headed data specification: " + spec)
    J = len(_data)
    s_Data[i] = np.tile(np.NaN, J-1)
    for j in range(1, J):
      _datum = _data[j][k]
      if isnumeric(_datum):
        s_Data[i][j-1] = float(_datum)
    s_Data[i] = nanravel(s_Data[i]) # remove NaNs
  for i in range(lend2-1, -1, -1): # remove empty data vectors
    if not(len(s_Data[i])):
      del s_Data[i]
      del _Labels[i]
  return _Labels, s_Data

def writeQfile(pf, _Data, _Labels = None): # using TSV format
  if _Labels is None:
    writeSVFile(pf, _Data)
    return _Data
  n = len(_Labels)
  Data = [[]] * n
  for i in range(n):
    Data[i] = [_Labels[i]]
    Data[i] += _Data[i]
  writeSVFile(pf, Data)
  return Data

class qmodl:
  path = None
  stem = None
  pmat = False
  mres = None
  defbw = None
  def __init__ (self, _X = None, _e = None):
    self.X = None
    self.maxn = 0
    self.NX = 0
    self.res = 0
    self.nlims = None
    self.iniLabels()
    self.iniHatValues()
    self.setRes()
    self.setData(_X, _e)
  def iniHatValues(self):
    self.paru = np.NaN
    self.parn = np.NaN
    self.parq = np.NaN
    self.hate = np.NaN
    self.hatq = np.NaN
    self.hatg = np.NaN
    self.hata = np.NaN
    self.hatr = np.NaN
    self.hatn = np.NaN
    self.hatl = np.NaN
    self.hatv = np.NaN
    self.hats = np.NaN
  def iniLabels(self):
    self.labels = None
    self.path = None
    self.stem = None
  def setRes(self, _nres = None, _ares = 1, _vres = 128, _sres = None, _rres = None, _qres = None):
    self.nres = _nres # if None - estimate
    self.ares = _ares
    self.vres = _vres
    self.sres = self.ares * self.vres if _sres is None else _sres
    if _rres is None:
      self.rres = max(self.sres, self.vres)
      if self.nres is not None: self.rres = max(self.rres, self.nres)
    else:
      self.rres = _rres
    self.qres = self.rres if _qres is None else _qres
    self.gres = self.vres
    self.vaRes = self.vres * self.ares
    self.svaRes = self.sres * self.vaRes
    self.modelBeta = self.ares > 1
  def readFile(self, _X = None, _e = None, spec = None):
    if _X is None: return
    if not(len(_X)): return
    ipdn, ipfn = os.path.split(_X)
    ipfs, ipfe = os.path.splitext(ipfn)
    ipfe = ipfe.lower()
    self.path = ipdn
    self.stem = ipfs
    self.labels, _X = readQfile(_X, spec)
    self.setData(_X, _e)
    return _X
  def setData(self, _X = None, _e = None):
    if _X is None:
      return
    elif type(_X) is str:
      self.readFile(_X, _e) # this function calls setData()
      return
    elif type(_X) is list:
      self.X = list2nan2D(_X)
    else:
      self.X = np.copy(_X)
    self.nx = nancnt(self.X, axis = 1)
    self.NX = len(self.nx)
    if self.labels is None: self.labels = np.arange(self.NX)
    self.mn = nanmean(self.X, axis = 1)
    self.Nx = np.sum(self.nx)
    self.mres = len(self.mn)
    self.vr = nanvar(self.X, axis = 1)
    self.vrvr = nanvrvr(self.X, axis = 1)
    self.maxn = np.max(nancnt(self.X, axis = 1))
    self.sd = np.sqrt(self.vr)
    self.se = self.sd / unzero(np.sqrt(np.array(self.nx, dtype = float)))
    self.cd = self.vr / self.mn
    self.cv = self.sd / self.mn
    self.minx = np.min(nanravel(self.X))
    self.maxx = np.max(nanravel(self.X))
    self.pol = 1.0 if abs(self.maxx) > abs(self.minx) else -1.0
    if self.pol > 0:
      self.amxm = self.maxx
      self.amnm = np.max(self.mn)
      self.amnn = np.min(self.mn)
      self.acdm = np.max(self.cd)
    else:
      self.amxm = self.minx
      self.amnm = np.min(self.mn)
      self.amnn = np.max(self.mn)
      self.acdm = np.min(self.cd)
    self.defbw = opthistbw(self.X)
    self.setNoise(_e)
  def setNoise(self, _e = None): # if an array of two, it defines a range
    mino = 1e-300
    if _e is None: _e = [0., np.inf]
    if isfloat(_e):
      if _e <= 0.:
        if self.stem is not None:
          i = self.stem.find("_e=")
          if i >= 0:
            _e = float(self.stem[i+3:])
      if _e <= 0.: raise ValueError("Baseline standard deviation must exceed 0.")
    if type(_e) is list:
      if len(_e) == 2:
        Xx = nanravel(self.X)
        y = Xx[Xx <= 0] if self.pol > 0 else Xx[Xx >= 0]
        if len(y) < 2:
          y = nanravel(self.X[np.argmax(abs(self.mn))])
          y = np.sort(np.fabs(y))
          y = y[:3]
        y = 1.2533 * np.mean(np.abs(y))
        y = max(_e[0], y)
        y = min(_e[1], y)
        _e = y
    self.e = _e
    self.e2 = self.e ** 2.0
    self.e2c = -1./(2. * self.e2 + mino)
    self.enc = 1./np.sqrt(2. * np.pi * self.e2 + mino)
    self.hate = self.e
    self.fitParabola()
  def fitParabola(self, alpha = 0.05, _parcv = None):
    sdc = 3.0
    mino = 1e-300
    if self.NX < 2: return
    self.parfit = parafit(self.mn, self.vr-self.e2, None, np.sqrt(self.vrvr))
    self.parmle = self.parfit[0]
    self.paru = self.parmle[0]
    self.parq = self.parmle[1]
    self.parn = -1./(self.paru+mino)
    self.parvar = np.array([np.inf, np.inf])
    if self.NX > 2:
      self.parvar = np.array((self.parfit[1][0,0], self.parfit[1][1,1]), dtype = float)
    self.parstd = np.sqrt(self.parvar)
    dof = self.NX-2
    self.parcid = np.sqrt(self.parvar / float(dof+mino) * stats.chi.ppf(0.5*alpha, dof)) # confidence interval delta
    self.parciu = self.parmle[0] + np.array( (-self.parcid[0], self.parcid[0]), dtype = float)
    self.parciq = self.parmle[1] + np.array( (-self.parcid[1], self.parcid[1]), dtype = float)
    self.parcv = _parcv
    if self.parcv is None:
      I = np.argsort(np.fabs(self.mn))
      y = np.empty(0, dtype = float)
      i = -1
      sdce = sdc * self.e
      while len(y) < 2 and i < self.NX - 1:
        i += 1
        y = nanravel(self.X[I[i]])
        y = y[y > sdce] if self.pol > 0 else y[y < -sdce]
        z = y[y < self.parq*2. - sdce] if self.pol > 0 else y[y > self.parq*2. + sdce]
        if len(z) < 2:
          z = y[y < self.parq*1.5] if self.pol > 0 else y[y > self.parq*1.5]
        if len(z) > 1:
          y = z
      self.parcv = iqr(y) / (1.349*nanmean(y)+mino)
      if self.parcv == 0.: self.parcv = 0.3 # wild guess if there are no failures.
    self.parcq = self.parq / (1. + self.parcv**2.0)
    self.parcu = self.paru / (1. + (0.5*self.parcv)**2.0)
    self.parcn = -1./ self.parcu
  def setLimits(self, _slims = None, _vlims = None, _alims = None, _nlims = None, _ngeo = None):
    # _ngeo = 0: linearly sampled with flat prior
    # _ngeo = 1: linearly sampled with reciprocal prior
    # _ngeo = 2: geometrically distributed with flat prior
    if _slims is None: _slims = [0.04, 0.96]
    if _vlims is None: _vlims = [0.05, 1.0]
    if _alims is None: _alims = [0.2, 20.0]
    if _ngeo is None: _ngeo = 0
    self.ngeo = _ngeo
    rhat = None
    if _nlims is None:
      rhat = self.amxm
      rmin = self.amnm
      rmax = rhat * 2.0
      qhat = self.acdm
      _nlims = [int(np.floor(rmin/qhat)), int(np.ceil(rmax/qhat))]
      _nlims[0] = min(_nlims[0], 1)
      _nlims[0] += not(_nlims[0])
      _nlims[1] = max(_nlims[1], _nlims[0]+1)
    if self.ngeo != 2: # if linearly sampled
      if self.nres is None:
        if isint(_nlims):  _nlims = [1, _nlims]
        self.nres = _nlims[1] - _nlims[0] + 1
    else: # geometrically sampled
      if rhat is not None: # if we estimated nlims, we need to revise lower limit
        _nlims = _nlims[1]
      if isint(_nlims):
        if self.nres is None: self.nres = int(np.log2(float(nmax)))
        nint = intspace(0, _nlims, self.nres, True)
        if not(len(nint)): nint = [1]
        _nlims = [nint[0], _nlims[1]]
    self.slims = _slims
    self.vlims = _vlims
    self.alims = _alims
    self.nlims = _nlims
    self.ilims = [0, self.nlims[1]]
  def setPriors(self, _pmat = None, _pgb = None):
    if _pmat is None: _pmat = False
    self.pmat = _pmat
    _n = np.array(numspace(self.nlims[0], self.nlims[1], self.nres, int(self.ngeo == 2)), dtype = int)
    self.Pr = discprob.mass(self.pmat)
    self.Pr.setx([self.mres, self.nres, self.ares, self.vres, self.sres], [0, 0, 1, 1, -2], self.mn, _n,
        self.alims, self.vlims, self.slims)

    #self.m = self.Pr.X[0]
    self.n = self.Pr.X[1]
    self.a = self.Pr.X[2]
    self.v = self.Pr.X[3]
    self.s = self.Pr.X[4]
    self.ires = self.n + 1
    self.v = self.v[::-1]
    self.Pr.X[3] = self.v
    priorSpec = [0,1,0,0,0] if self.ngeo == 1 else [0,0,0,0,0]
    self.Pr.setUniPrior(priorSpec)

    # Gamma depends only on v

    self.gv = 1./(self.v**2.)
    self.G = [[]] * self.nres
    self.Gs = [[]] * self.nres #gs
    for i in range(self.nres):
      self.G[i] = np.tile(self.gv.reshape(1, self.vres, 1, 1), (self.ares, 1, self.sres, self.ires[i]))
      self.Gs[i] = np.tile(self.gv.reshape(1, self.vres, 1, 1), (self.ares, 1, self.sres, self.n[i]))
      if self.pmat:
        self.G[i] = np.matrix(self.G[i].reshape((self.svaRes, self.ires[i])))
        self.Gs[i] = np.matrix(self.Gs[i].reshape((self.svaRes, self.n[i])))

    # Bernoulli model

    self.I = [[]] * self.nres
    self.Is = [[]] * self.nres
    for i in range(self.nres):
      self.I[i] = np.arange(self.ires[i], dtype = int)
      self.Is[i] = np.array(self.I[i][1:], dtype = float)

    # Gamma_i depends on self.n and self.Is (but not on self.mn)

    self.GI = [[]] * self.nres
    for i in range(self.nres):
      self.GI[i] = self.gv.reshape((self.gres, 1)) * self.Is[i].reshape((1, self.n[i]))
      if self.pmat: self.GI[i] = np.matrix(self.GI[i].reshape((self.gres, self.n[i])))

    self.Gi = [[]] * self.nres # g
    self.Gd = [[]] * self.nres # g-1.
    self.Gl = [[]] * self.nres # glnf(g)
    self.Z0 = [[]] * self.nres
    for i in range(self.nres):
      _Gi = np.array(self.GI[i]) if self.pmat else self.GI[i]
      _Gd = _Gi - 1.
      _Gl = discprob.glnf(_Gi)
      self.Gi[i] = np.tile(_Gi.reshape((1, self.gres, 1, self.n[i])), (self.ares, 1, self.sres, 1))
      self.Gd[i] = np.tile(_Gd.reshape((1, self.gres, 1, self.n[i])), (self.ares, 1, self.sres, 1))
      self.Gl[i] = np.tile(_Gl.reshape((1, self.gres, 1, self.n[i])), (self.ares, 1, self.sres, 1))
      self.Z0[i] = np.zeros((self.ares, self.gres, self.sres, self.n[i]), dtype = float)
      if self.pmat:
        self.Gi[i] = np.matrix(self.Gi[i].reshape((self.svaRes, self.n[i])))
        self.Gd[i] = np.matrix(self.Gd[i].reshape((self.svaRes, self.n[i])))
        self.Gl[i] = np.matrix(self.Gl[i].reshape((self.svaRes, self.n[i])))
        self.Z0[i] = np.matrix(self.Z0[i].reshape((self.svaRes, self.n[i])))

    # Binomial coefficients depend on self.n, self.s, and self.i (beta coefficients also depend on self.a)

    self.IS, self.C = None, None
    self.A , self.S , self.B  = None, None, None

    self.Ci = [[]] * self.nres
    self.Cf = [[]] * self.nres
    self.Cs = [[]] * self.nres
    self.Ss = [[]] * self.nres

    if not(self.modelBeta):
      self.IS = [[]] * self.nres
      self.S = [[]] * self.nres
      self.C = [[]] * self.nres
      for i in range(self.nres):
        self.IS[i] = np.tile(self.I[i].reshape(1, self.ires[i]), (self.sres, 1))
        self.S[i] = np.tile(self.s.reshape(self.sres, 1), (1, self.ires[i]))
        self.C[i] = stats.binom.pmf(self.IS[i], self.n[i], self.S[i])
        self.Ci[i] = np.tile(self.C[i].reshape((1, 1, self.sres, self.ires[i])), (self.ares, self.vres, 1, 1))
        self.Cf[i], self.Cs[i] = self.Ci[i][:, :, :, 0], self.Ci[i][:, :, :, 1:]
        self.Ss[i] = np.tile(self.S[i].reshape(1, 1, self.sres, self.ires[i]), (self.ares, self.vres, 1, 1))[:, :, :, 1:]
    else:
      self.A = np.tile(self.a.reshape(self.ares, 1), (1, self.sres))
      self.S = np.tile(self.s.reshape(1, self.sres), (self.ares, 1))
      self.B = self.A/self.S - self.A
      self.C = discprob.betanompmf(self.n, self.A, self.B, _pgb)
      for i in range(self.nres):
        self.Ci[i] = np.tile(self.C[i].reshape((self.ares, 1, self.sres, self.ires[i])), (1, self.vres, 1, 1))
        self.Cf[i], self.Cs[i] = self.Ci[i][:, :, :, 0], self.Ci[i][:, :, :, 1:]
        self.Ss[i] = np.tile(self.s.reshape(1, 1, self.sres, 1), (self.ares, self.vres, 1, self.ires[i]))[:, :, :, 1:]

    if self.pmat:
      for i in range(self.nres):
        #self.C[i] = np.matrix(self.C[i])
        self.Ci[i] = np.matrix(self.Ci[i].reshape(self.svaRes, self.ires[i]))
        self.Cf[i] = np.matrix(self.Cf[i].reshape(self.svaRes, 1))
        self.Cs[i] = np.matrix(self.Cs[i].reshape(self.svaRes, self.n[i]))
        self.Ss[i] = np.matrix(self.Ss[i].reshape(self.svaRes, self.n[i]))

    # Qs depends on self.mn, self.n, and self.s , and Ls additionally on self.g
    # (admittedly q with self.ares*self.gres*self.n[i]-fold,
    #         and l with self.ares*self.n[i]-fold redundancy)

    _qmin, _qmax = np.inf, -np.inf
    _lmin, _lmax = np.inf, -np.inf
    self.Qs = [[]] * self.mres
    self.Ls = [[]] * self.mres # l
    self.Lg = [[]] * self.mres # -g*log(l) - glnf(g)
    self.Lr = [[]] * self.mres # -1./l
    if _pgb is not None: _pgb.init("Conditioning priors", self.mres)
    for h in range(self.mres):
      if _pgb is not None: _pgb.set(h, True)
      self.Qs[h] = [[]] * self.nres
      self.Ls[h] = [[]] * self.nres
      self.Lg[h] = [[]] * self.nres
      self.Lr[h] = [[]] * self.nres
      for i in range(self.nres):
        self.Qs[h][i] = (np.fabs(self.mn[h]) / self.Ss[i]) / float(self.n[i])
        self.Ls[h][i] = self.Qs[h][i] / self.Gs[i]
        self.Lr[h][i] = -1./self.Ls[h][i]
        self.Lg[h][i] =  - np.multiply(self.Gi[i], np.log(self.Ls[h][i])) - self.Gl[i]
        _qmin = min(_qmin, self.Qs[h][i].min())
        _qmax = max(_qmax, self.Qs[h][i].max())
        _lmin = min(_lmin, self.Ls[h][i].min())
        _lmax = max(_lmax, self.Ls[h][i].max())
        if self.pmat:
          self.Qs[h][i] = np.matrix(self.Qs[h][i].reshape((self.svaRes, self.n[i])))
          self.Ls[h][i] = np.matrix(self.Ls[h][i].reshape((self.svaRes, self.n[i])))
          self.Lr[h][i] = np.matrix(self.Lr[h][i].reshape((self.svaRes, self.n[i])))
          self.Lg[h][i] = np.matrix(self.Lg[h][i].reshape((self.svaRes, self.n[i])))
    if _pgb is not None: _pgb.reset()


    '''
    if self.pmat: # mould matrices into efficient shapes
      for h in range(self.mres):
        for i in range(self.nres):
          self.Lr[h][i] = self.Lr[h][i].T
          self.Lg[h][i] = self.Lg[h][i].T
      for i in range(self.nres):
        self.Gd[i] = self.Gd[i].T

    '''

    self.qlims = [self.pol*_qmin, self.pol*_qmax]
    self.llims = [self.pol*_lmin, self.pol*_lmax]
    self.rlims = [self.qlims[0]*float(self.nlims[0]), self.qlims[1]*float(self.nlims[1])]
  def calcPosts(self, parallel = None, pgb = None):
    if parallel is None:
      parallel = MAXTHREADS > 1
    self.vanRes = self.vaRes * self.nres
    self.LL = np.log(self.Pr.P)
    self.PP = [[]] * self.mres
    self.LLm = np.empty(self.mres, dtype = float)
    if parallel:
      self.preFetchParallel(pgb)
      self.calcPostsParallel(pgb)
    else:
      self.preFetchSerial(pgb)
      self.calcPostsSerial(pgb)
    self.PP = np.array(self.PP).reshape(self.Pr.nX)
    self.P = discprob.mass(self.PP, self.Pr.X, self.pmat)
    self.PMNAVS = self.P.copy()
    self.calcCondPosts(pgb)
    self.calcJointPost()
    self.calcMargPosts()
    self.calcHatValues()
  def preFetchSerial(self, pgb = None):
    mino = 1e-300
    if pgb is not None: pgb.init("Prefetching convolutions", self.Nx+2)
    c = -1
    self.Gm = [[]] * self.mres
    for k in range(self.mres):
      Xk = self.pol * nanravel(self.X[k])
      self.Gm[k] = [[]] * self.nx[k]
      for j in range(self.nx[k]):
        c += 1
        if pgb is not None: pgb.set(c)
        x = Xk[j]
        self.Gm[k][j] = [[]] * self.nres
        for i in range(self.nres):
          if x <= 0.:
            self.Gm[k][j][i] = self.Z0[i]
          else:
            #Gm = discprob.gampdf(x, self.Gi[i], self.Ls[k][i])
            #y = np.exp(- g * np.log(l) - x/l - sp.special.gammaln(g) + (g-1.0)*np.log(x) )
            self.Gm[k][j][i] = np.exp(self.Lg[k][i] + x*self.Lr[k][i] + self.Gd[i] * np.log(x))
    if pgb is not None: pgb.reset()
  def calcPostsSerial(self, pgb = None):
    mino = 1e-300
    if pgb is not None: pgb.init("Calculating posteriors", self.Nx+2)
    c = -1
    for k in range(self.mres):
      Xk = self.pol * nanravel(self.X[k])
      for j in range(self.nx[k]):
        c += 1
        if pgb is not None: pgb.set(c)
        x = Xk[j]
        ri = k * self.vanRes
        rj = ri + self.vaRes
        #L0 = stats.norm.pdf(x, 0., self.e)
        L0 = self.enc * np.exp(x*x*self.e2c)
        for i in range(self.nres):
          L = L0 * self.Cf[i]
          if x >= 0.:
            if self.pmat:
              #L += np.sum(np.multiply(self.Cs[i], Gm), axis = -1)
              L += diag_inner(self.Cs[i], self.Gm[k][j][i])
            else:
              L += diag_inner(self.Cs[i], self.Gm[k][j][i])
              #L += np.sum(np.multiply(self.Cs[i], Gm), axis = -1)
          if self.pmat:
            self.LL[ri:rj, :] += np.log(L+mino).reshape((self.vaRes, self.sres))
            ri, rj = rj, rj+self.vaRes
          else:
            self.LL[k, i, :, :, :] += np.log(L + mino)
      if self.pmat:
        ri = k * self.vanRes
        rj = ri + self.vanRes
        LLk = self.LL[ri:rj,:]
        self.LLm[k] = LLk.max()
      else:
        LLk = self.LL[k, :, :, :, :]
        self.LLm[k] = LLk.max()
      self.PP[k] = np.exp(LLk-self.LLm[k])
      self.PP[k] /= (self.PP[k].sum()+mino)
    if pgb is not None: pgb.reset()
  def preFetchParallel(self, pgb = None):
    mino = 1e-300
    if pgb is not None: pgb.init("Prefetching convolutions", self.Nx+2)
    c = -self.nx[0]
    self.Gm = [[]] * self.mres
    for k in range(self.mres):
      x = self.pol * nanravel(self.X[k])
      xres = self.nx[k]
      c += xres
      if pgb is not None: pgb.set(c, True)
      self.Gm[k] = [[]]*self.nres
      for j in range(self.nres):
        Lgkj = self.Lg[k][j]
        Lrkj = self.Lr[k][j]
        Gdj = self.Gd[j]
        self.Gm[k][j] = np.empty((xres, self.ares, self.vres, self.sres, self.n[j]), dtype = float)
        if Parallel is None:
          for i in range(xres):
            xi = x[i]
            if xi <= 0.:
              self.Gm[k][j][i] = self.Z0[j]
            else:
              lxi = np.log(xi)
              #Gm = discprob.gampdf(x, self.Gi[i], self.Ls[k][i])
              #y = np.exp(- g * np.log(l) - x/l - sp.special.gammaln(g) + (g-1.0)*np.log(x) )
              self.Gm[k][j][i] = np.exp(Lgkj + Lrkj*xi + Gdj*lxi)
        else:
          maxthreads = max(MAXTHREADS, xres)
          Gmkj = Parallel(n_jobs = maxthreads, backend="threading")(
              delayed(ParExpSum)(i, x, Lgkj, Lrkj, Gdj, self.Z0[j]) for i in range(xres))
          self.Gm[k][j] = np.array(Gmkj)
    if pgb is not None: pgb.reset()
  def calcPostsParallel(self, pgb = None):
    mino = 1e-300
    if pgb is not None: pgb.init("Calculating posteriors", self.Nx+2)
    c = -self.nx[0]
    if self.pmat:
      raise TypeError("Parallel computation not supported for self.pmat=True")
    else:
      for k in range(self.mres):
        x = self.pol * nanravel(self.X[k])
        _x = np.fabs(x)+mino
        _lx = np.log(_x)
        xf = x <= 0.
        xres = self.nx[k]
        c += xres
        if pgb is not None: pgb.set(c, True)
        L0 = self.enc * np.exp(np.multiply(x, x)*self.e2c)
        for j in range(self.nres):
          Gmkj = self.Gm[k][j]
          nj = self.n[j]
          Gdj = self.Gd[j]
          Csj = self.Cs[j]
          Csf = self.Cf[j]
          _X = np.tile(_x.reshape((1, xres)), (nj, 1))
          _LX = np.tile(_lx.reshape((1, xres)), (nj, 1))
          for i in range(self.ares):
            for h in range(self.vres):
              for f in range(self.sres):
                #t0 = time.time()
                L = Csf[i][h][f] * L0
                _Cs = Csj[i][h][f].reshape((1, nj))
                _Gm = Gmkj[:,i,h,f].reshape((xres, nj))
                #t1 = time.time()
                #t2 = time.time()
                _Gm[xf, :] = 0.
                L += np.inner(_Cs, _Gm)[0]
                #t3 = time.time()
                self.LL[k, j, i, h, f] += np.sum(np.log(L+mino))
                #print(nj, 1000000.*(t1-t0), 1000000.*(t2-t1), 1000000.*(t3-t2))
        self.LLm[k] = self.LL[k].max()
        self.PP[k] = np.exp(self.LL[k]-self.LLm[k])
        self.PP[k] /= (self.PP[k].sum()+mino)
      if pgb is not None: pgb.reset()
  def calcCondPosts(self, pgb = None):
    mino = 1e-300
    self.g = np.sort(self.gv)
    PMP = np.zeros(self.mres, dtype = float)
    for i in range(self.mres):
      for j in range(self.mres):
        PMP[i] += discprob.zpdf(self.mn[i], self.mn[j], self.se[j]**2.)
    PMP /= unzero(PMP.sum())
    self.PM = discprob.mass(PMP, [self.mn])
    self.PMRQGA = discprob.mass(self.pmat)
    self.PMRQGA.setx([self.mres, self.rres, self.qres, self.gres, self.ares], [0, 1, 1, 0, 0], self.mn, self.rlims,
        self.qlims, self.g, self.a)
    self.PMRQGA.setUniPrior()
    self.r, self.q = self.PMRQGA.X[1], self.PMRQGA.X[2]
    if pgb is not None: pgb.init("Conditioning posteriors", self.mres)
    for i in range(self.mres):
      if pgb is not None: pgb.set(i)
      pr = self.PMRQGA.slice(0, i)
      lh = self.PMNAVS.slice(0, i)
      lhR = self.mn[i] / (lh.repa(3)+mino)
      lhQ = lhR / (lh.repa(0)+mino)
      lhG = 1.0 / (np.power(lh.repa(2), 2.) + mino)
      lhA = lh.repa(1)
      pr.calcPost(lh, lhR, lhQ, lhG, lhA)
      pr.normalise()
      self.PMRQGA.setPslice(0, i, pr.P / unzero(self.PM.P[i]))
    if pgb is not None: pgb.reset()
  def calcJointPost(self):
    mino = 1e-300
    inc = 1e-15
    _PMRQGA  = self.PMRQGA.copy()
    done = False
    while not(done): # horrible hack to prevent crashes from highly discordant data: crap in -> crap out
      self.PRQGA = _PMRQGA.multiply(0) # don't post-normalise because we've improved sensitivity using PM
      if self.PRQGA.P.max() > mino:
        done = True
      else:
        _PMRQGA.P += inc
        inc *= 2.
    self.PRQGA.normalise()
  def calcMargPosts(self):
    mino = 1e-300
    self.PMQGA, self.PMRGA = self.PMRQGA.marginalise(1), self.PMRQGA.marginalise(2)
    self.PMRQA, self.PMRQG = self.PMRQGA.marginalise(3), self.PMRQGA.marginalise(4)

    self.PMQG, self.PMRG, self.PMRQ = self.PMRQG.marginalise(1), self.PMRQG.marginalise(2), self.PMRQG.marginalise(3)
    self.PMQA, self.PMRA, self.PMGA = self.PMRQA.marginalise(1), self.PMRQA.marginalise(2), self.PMRGA.marginalise(1)

    self.PQGA, self.PRGA = self.PRQGA.marginalise(0), self.PRQGA.marginalise(1)
    self.PRQA, self.PRQG = self.PRQGA.marginalise(2), self.PRQGA.marginalise(3)

    self.PGA, self.PQA, self.PQG = self.PQGA.marginalise(0), self.PQGA.marginalise(1), self.PQGA.marginalise(2)
    self.PQG, self.PRG, self.PRQ = self.PRQG.marginalise(0), self.PRQG.marginalise(1), self.PRQG.marginalise(2)

    self.PQ, self.PR = self.PRQ.marginalise(0), self.PRQ.marginalise(1)
    self.PA, self.PG = self.PGA.marginalise(0), self.PGA.marginalise(1)

    self.PV = self.PG.copy()
    self.PV.setX([1./np.sqrt(self.PG.X[0])])

    self.PN = discprob.mass(self.pmat)
    self.PN.setX([self.n])
    self.PN.setUniPrior()
    lh = self.PRQGA.copy()
    lhN = np.fabs(lh.repa(0) / lh.repa(1)+mino)
    self.PN.calcPost(lh, lhN)
    self.PN.normalise()

    self.PMS = discprob.mass(self.pmat)
    self.PMS.setx([self.mres, self.sres], [0, 0], self.mn, self.s)
    self.PMS.setUniPrior()
    for i in range(self.mres):
      pr = discprob.mass(self.PMS.P[i, :], self.PMS.X[1], self.pmat)
      lhr = np.fabs(self.mn[i] / (self.PR.X[0])+mino)
      if self.pmat: lhr = np.matrix(lhr)
      pr.calcPost(self.PR, lhr)
      self.PMS.P[i, :] = pr.P / (pr.P.sum() + mino)

  def calcHatValues(self):
    mino = 1e-300
    self.absr, self.absq = np.fabs(self.r), np.fabs(self.q)
    self.labsr, self.labsq, self.labsg, self.labsa = np.log10(self.absr), np.log10(self.absq), np.log10(self.g), np.log10(self.a)
    self.hatr = float(self.PR.sample(1))
    self.hatq = float(self.PQ.sample(1))
    self.hatg = float(self.PG.sample(1))
    self.hata = float(self.PA.sample(1)) if self.modelBeta else np.NaN
    self.hatn = self.hatr / (self.hatq + mino)
    self.hatl = self.hatq / (self.hatg + mino)
    self.hatv = 1.0 / (np.sqrt(self.hatg) + mino)
    self.hats = np.array([mn_ / (self.hatr+mino) for mn_ in self.mn], dtype = float)
    self.hatvr= QLV(self.mn, self.hate, self.hatn, self.hatg, self.hatl, self.hata)
  def archive(self, _arch = None):
    # this function does different things depending on what _arch is and whether we have data (X)
    #
    # If we have data (self.X is not None):
    # _arch is None writes results to self.arch
    # _arch is string writes results to self.arch which is written to filename _arch
    # otherwise self.arch = _arch, and processed
    #
    # If we have no data (self.X is None)
    # _arch is None returns the function doing nothing
    # _arch is string reads data from filename _arch to restore class variables, and process
    # otherwise _arch = self.arch, and processed

    process = False

    if self.X is None:
      if _arch is None: return
      process = True
      self.arch = readPickle(_arch) if type(_arch) is str else _arch
      if type(_arch) is str:
        ipdn, ipfn = os.path.split(_arch)
        ipfs, ipfe = os.path.splitext(ipfn)
        ipfe = ipfe.lower()
        self.path = ipdn
        self.stem = ipfs
    else:
      if _arch is None:
        self.arch = [self.X, self.e, self.n, self.s, self.PMRQGA]
        return self.arch
      elif type(_arch) is str: # we split this to prevent recursive processing
        self.arch = self.archive()
        writePickle(_arch, self.arch)
      else:
        self.arch = _arch
        process = True

    if not(process): return

    # we have to repopulate the class members with self.arch
    [self.X, self.e, self.n, self.s, self.PMRQGA] = self.arch
    self.setData(self.X, self.e)
    self.mn, self.r, self.q, self.g, self.a = self.PMRQGA.X
    self.nres, self.sres = len(self.n), len(self.s)
    self.mres, self.qres, self.rres = len(self.mn), len(self.q), len(self.r)
    self.gres, self.ares, self.vres = len(self.g), len(self.a), len(self.g)
    self.modelBeta = self.ares > 1
    self.calcJointPost()
    self.calcMargPosts()
    self.calcHatValues()

class Qmodl (qmodl): # A pyqtgraph front-end for qmodl
  defxy = [720, 540]
  form = None
  area = None
  dgei = None
  PQ = None
  PA = None
  PR = None
  PG = None
  PV = None
  PN = None
  def openFile(self, spec = True):
    self.Base = lbw.LBWidget(None, None, None, 'base')
    self.pf = self.Base.dlgFileOpen("Open File", "", "Tab-delimited (*.tdf *.tab *.tsv);;Excel file (*.xlsx *.xls);;All (*.*)")
    if self.pf is None: return
    rf =  self.readFile(self.pf, None, spec)
    if rf is not None:
      self.SetNoise()
  def SetNoise(self):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.EDx = lbw.LBWidget(self.Dlg, "Enter baseline standard deviation: ", 1, 'edit', str(self.e))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.blhelp)
    self.BBx.Widgets[1].connect("btndown", self.SetNoiseOK)
    self.Box.add(self.EDx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def blhelp(self):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[0]+MANSUFF)
  def SetNoiseOK(self):
    self.uie = self.EDx.retData()
    self.Dlg.close()
    self.uie = None if not(isnumeric(self.uie)) else float(self.uie)
    del self.Dlg
    self.setData(self.X, self.uie)
    self.iniGUI().show()
  def Archive(self):
    self.Base = lbw.LBWidget(None, None, None, 'base')
    readData = self.X is None
    if readData: # read
      self.pf = self.Base.dlgFileOpen("Open File", "", "Results file (*.pkl);;All (*.*)")
    else:        # write
      if self.path is None or self.stem is None:
        self.pf = self.Base.dlgFileSave("Save File", "results.pkl", "Results file (*.pkl);;All (*.*)")
      else:
        opdn, opfs = self.path, self.stem
        _opdn = opdn.replace('analyses', 'results')
        if os.path.exists(_opdn): opdn = _opdn
        defop = opdn + '/' + opfs + ".pkl"
        self.pf = self.Base.dlgFileSave("Save File", defop, "Results file (*.pkl);;All (*.*)")
    if self.pf is None: return
    if not(len(self.pf)): return
    self.archive(self.pf)
    if readData:
      self.iniGUI().show()
  def plotMoments(self, _plot = None):
    mino = 1e-300
    axc = 1.05;
    rxc = 0.99
    num = 1000;
    newplot = False
    if _plot is None:
      newplot = True
      _plot = pq.graph()
    errd = np.sqrt(self.vrvr)
    maxabsr = np.max(np.fabs(self.mn))
    if not(np.isnan(self.paru)):
      if self.parn > 0.:
        _maxabsr = self.parn * np.fabs(self.parq)
        if _maxabsr < 2.*maxabsr:
          maxabsr = _maxabsr
    if not(np.isnan(self.hatn)):
      _maxabsr = self.hatn * np.fabs(self.hatq)
      maxabsr = max(maxabsr, _maxabsr)
    _hatmn = self.pol * np.linspace(mino, maxabsr, num)
    maxhatv = np.max(self.vr+errd)
    Hatvr = [None, None]
    if not(np.isnan(self.paru)):
      _hatvr = parafun(_hatmn, self.paru, self.parq, self.hate**2.0)
      maxhatv = max(maxhatv, np.max(_hatvr))
      Hatvr[0] = pg.PlotCurveItem(_hatmn, _hatvr, pen={'color':'r','width':1})
    if not(np.isnan(self.hatn)):
      _hatr = float(self.hatq * int(np.round((self.hatn))))
      hatmn_ = _hatmn[_hatmn <= _hatr] if self.pol > 0. else _hatmn[_hatmn >= _hatr]
      hatvr_ = QLV(np.abs(hatmn_), self.hate, self.hatn, self.hatg, np.abs(self.hatl), self.hata)
      maxhatv = max(maxhatv, np.max(hatvr_))
      Hatvr[1] = pg.PlotCurveItem(hatmn_, hatvr_, pen={'color':'g','width':1})
    xLims = np.array((0., self.pol*maxabsr), dtype = float)
    yLims = np.array((0., maxhatv), dtype = float)
    Lims = pg.PlotCurveItem(xLims, yLims, pen=pg.mkPen(None))
    _plot.add(Lims)
    err = pg.ErrorBarItem(x=self.mn, y=self.vr, top=errd, bottom=errd)
    _plot.add(err)
    _scat = _plot.scat()
    _scat.addPoints(self.mn, self.vr)
    for hatvr in Hatvr:
      if hatvr is not None: _plot.add(hatvr)
    _plot.setLabel('left', "Variance")
    _plot.setLabel('bottom', "Mean")
    if newplot: return _plot
  def plotMarg1D(self, i = None, _gbox = None): # I: 0 = R, 1 = Q, 2 = V, 3 = A, 4 = v, 5 = n
    if np.isnan(self.hatn): return
    _P = [self.PR, self.PQ, self.PG, self.PA, self.PV, self.PN]
    _X = ['r', 'q', 'gamma', 'alpha', 'CV', 'n']
    if self.pol < 0.:
      _X[0] = "".join(("-", _X[0]))
      _X[1] = "".join(("-", _X[1]))
    _X[0] = "".join(("log10(", _X[0], ")"))
    _X[1] = "".join(("log10(", _X[1], ")"))
    _X[2] = "".join(("log10(", _X[2], ")"))
    if i is None and _gbox is None:  # New figure
      _form = pq.BaseFormClass()
      _area = pq.area()
      _form.setCentralWidget(_area)
      _dock0 = pq.dock('')
      _dock1 = pq.dock('')
      _dock2 = pq.dock()
      _area.add(_dock0)
      _area.add(_dock1)
      _gbox0 = _dock0.addGbox()
      _gbox1 = _dock1.addGbox()
      _graph = [[]] * 6
      _gb = [_gbox0, _gbox0, _gbox0, _gbox1, _gbox1, _gbox1]
      for i in range(6):
        if i != 3 or self.modelBeta:
          _graph[i] = self.plotMarg1D(i, _gb[i])
      _area.add(_dock2)
      if self.dgei is not None:
        _bbox = _dock2.addBbox()
        _bbox.addButton()
        _bbox.setIconSize(0, QtCore.QSize(1,1))
        _bbox.setText(0, 'Export')
        _bbox.Connect(0, self.ExpMarg1D)
      return _form
    _plot = pq.graph(parent=_gbox)
    P_ = _P[i]
    if P_ is not None:
      x = nanravel(P_.X[0])
      y = nanravel(P_.P)
      if i < 2: x *= self.pol
      if i < 3: x = np.log10(np.abs(x))
      if i < 5:
        _mp = pg.PlotCurveItem(x, y, pen={'color':'w','width':1})
      else:
        xLims = [0., np.max(x)]
        yLims = [0., np.max(y)]
        Lims = pg.PlotCurveItem(xLims, yLims, pen=pg.mkPen(None))
        _plot.add(Lims)
        _mp = _plot.bar(x, y, _pen={'color':'w','width':1})
      _plot.add(_mp)
      _plot.setLabel('left', "Probability")
      _plot.setLabel('bottom', _X[i])
  def plotxyZ(self, _x, _y, _Z, _vbox = None):
    _mesh = pq.surf(_x, _y, _Z, shader='normalColor')
    _vbox = pq.BaseVboxClass() if _vbox is None else _vbox
    _vbox.addItem(_mesh)
    return _vbox
  def plotMarg2D(self, spec = 0, cond = False, _vbox = None):
    # spec: 0 = PRQ, 1 = PQG, 2 = PRG, 3 = PGA
    # cond: False = joint, True = all cond, otherwise by index
    JP = [self.PRQ, self.PQG, self.PRG, self.PGA]
    CP = [self.PMRQ, self.PMQG, self.PMRG, self.PMGA]
    if spec == 0: self.x3d, self.y3d = self.labsr, self.labsq
    if spec == 1: self.x3d, self.y3d = self.labsq, self.labsg
    if spec == 2: self.x3d, self.y3d = self.labsr, self.labsg
    if spec == 3: self.x3d, self.y3d = self.labsg, self.labsa
    if _vbox is None:
      _form = pq.BaseFormClass()
      _area = pq.area()
      _form.setCentralWidget(_area)
      if not(cond):
        self.P3d = JP[spec]
        _dock = pq.dock('')
        _vbox = _dock.addVbox()
        _area.add(_dock)
        self.plotMarg2D(spec, False, _vbox)
      else:
        self.P3d = CP[spec]
        _dock = [[]] * self.mres
        _vbox = [[]] * self.mres
        for i in range(self.mres):
          _dock[i] = pq.dock('')
          if not(i):
            _area.add(_dock[i])
          else:
            _area.add(_dock[i], 'right', _dock[i-1])
          _vbox[i] = _dock[i].addVbox()
          self.plotMarg2D(spec, i, _vbox[i])
      _Dock = pq.dock()
      _area.add(_Dock)
      _bbox = _Dock.addBbox()
      _bbox.addButton()
      _bbox.setIconSize(0, QtCore.QSize(1,1))
      _bbox.setText(0, 'Export')
      _bbox.Connect(0, self.ExpMarg2D)
      return _form
    else:
      if type(cond) is bool:
        return self.plotxyZ(self.x3d, self.y3d, JP[spec].P, _vbox)
      else:
        return self.plotxyZ(self.x3d, self.y3d, CP[spec].P[cond], _vbox)
  def plotHist(self, i = None, bw = None, _gbox0 = None, _gbox1 = None):
    if bw is None: bw = self.defbw
    if i is None and _gbox0 is None:  # New figure
      _form = pq.BaseFormClass()
      _area = pq.area()
      _form.setCentralWidget(_area)
      _dock0 = pq.dock('')
      _dock1 = pq.dock('')
      _dock2 = pq.dock()
      _area.add(_dock0)
      _gbox0 = _dock0.addGbox()
      _gbox1 = None
      if not(np.isnan(self.hatn)):
        _area.add(_dock1)
        _gbox1 = _dock1.addGbox()
      plothist = [[]] * self.mres
      self.xHist, self.yHist = [[]] * self.mres, [[]] * self.mres
      self.xQLF, self.yQLF = [[]] * self.mres, [[]] * self.mres
      for i in range(self.mres):
        self.xHist[i], self.yHist[i], self.xQLF[i], self.yQLF[i] = self.plotHist(i, bw, _gbox0, _gbox1)
      _area.add(_dock2)
      _bbox = _dock2.addBbox()
      _bbox.addButton()
      _bbox.setIconSize(0, QtCore.QSize(1,1))
      _bbox.setText(0, 'Export')
      _bbox.Connect(0, self.ExpHist)
      return _form
    num = 1000
    Xi = nanravel(self.X[i])
    x = binspace(self.minx, self.maxx, bw)
    hx = np.linspace(self.minx, self.maxx, num)
    hy = None
    dh = x[1] - x[0]
    h = freqhist(Xi, x)
    maxh = np.max(h)
    if not(np.isnan(self.hatn)):
      hatl = self.hatl
      if self.pol < 0. and hatl > 0.:
        hatl *= -1.
      hy = QLF(hx, self.hate, np.round(self.hatn), self.hatg, hatl, self.hata, self.hats[i])
      hy *= self.nx[i] * dh
      maxh = max(maxh, np.max(hy))
    _plot0 = pq.graph(parent=_gbox0)
    Lims = pg.PlotCurveItem([self.minx, self.maxx], [0., float(maxh)], pen=pg.mkPen(None))
    _plot0.add(Lims)
    barch = _plot0.bar(x, h, _pen={'color':'w','width':1})
    if not(np.isnan(self.hatn)):
      qlfdn = pg.PlotCurveItem(hx, hy, pen={'color':'w','width':1})
      _plot0.add(qlfdn)
    if _gbox1 is None or np.isnan(self.hatn): return x, h, hx, hy
    _plot1 = pq.graph(parent=_gbox1)
    if self.dgei is None:
      _mp = pg.PlotCurveItem(self.s, self.PMS.P[i], pen={'color':'w','width':1})
      _plot1.add(_mp)
      _plot1.setLabel('left', "Probability")
      _plot1.setLabel('bottom', "Probability")
    return x, h, hx, hy
  def plotHatValues(self, _tabl = None):
    tabl_ = pq.tabl() if _tabl is None else _tabl
    tabl_.setSortingEnabled(False)
    _results = []
    _results.append(['N', str(int(self.nx.min())) + "-" + str(int(self.nx.max()))])
    _results.append(['epsilon', str(float(self.hate))])
    _results.append(['MPFA_Q', str(float(self.parq))])
    _results.append(['MPFA_N', str(float(self.parn))])
    if not(np.isnan(self.hatn)):
      if self.dgei is None:
        _results.append(['Res.', str(self.sres) + ", " + str(self.vres) + ", " + str(self.ares) + ", " + str(self.nres)])
      else:
        _results.append(['Res.', str(self.nres) + ", " + str(self.qres) + ", " + str(self.gres)])
      _results.append(['r', str(float(self.hatr))])
      _results.append(['q', str(float(self.hatq))])
      _results.append(['gamma', str(float(self.hatg))])
      _results.append(['alpha', str(float(self.hata))])
      _results.append(['lambda', str(float(self.hatl))])
      _results.append(['v', str(float(self.hatv))])
      _results.append(['n', str(float(self.hatn))])
      _results.append(['s', self.hats])
    tabl_.setData(_results)
    return tabl_
  def iniForm(self, _form = None):
    if self.form is None:
      if _form is None: _form = pq.BaseFormClass()
      self.form = _form
    if self.area is None:
      self.area = pq.area()
    try:
      self.form.setCentralWidget(self.area)
    except AttributeError:
      self.form.setWidget(self.area)
  def iniGUI(self):
    self.out = []
    self.iniForm()
    self.dock0 = pq.dock('')
    self.dock1 = pq.dock('')
    self.dock2 = pq.dock()
    self.area.add(self.dock0)
    self.area.add(self.dock1, 'right', self.dock0)
    self.area.add(self.dock2, 'bottom')
    self.gbox0 = self.dock0.addGbox()
    self.gbox1 = self.dock1.addGbox()
    self.bbox = self.dock2.addBbox()
    self.grap = pq.graph(parent=self.gbox0)
    self.plotMoments(self.grap)
    self.tabl = pq.tabl()
    self.tabl.setParent(self.gbox1)
    self.plotHatValues(self.tabl)
    self.bbox.addButton()
    self.bbox.setIconSize(0, QtCore.QSize(1,1))
    self.bbox.setText(0, 'Histograms')
    self.bbox.Connect(0, self.SetBins)
    if np.isnan(self.hatn):
      self.bbox.addButton()
      self.bbox.setIconSize(1, QtCore.QSize(1,1))
      self.bbox.setText(1, 'Run original BQA')
      self.bbox.Connect(1, self.SetRes)
      self.bbox.addButton()
      self.bbox.setIconSize(1, QtCore.QSize(1,1))
      self.bbox.setText(2, 'Run BQA DGEI')
      self.bbox.Connect(2, self.RunDGEI)
    else:
      self.bbox.addButton()
      self.bbox.setIconSize(1, QtCore.QSize(1,1))
      self.bbox.setText(1, 'Marginal 1D')
      self.bbox.Connect(1, self.PlotMarg1D)
      self.bbox.addButton()
      if self.dgei is None:
        self.bbox.setIconSize(2, QtCore.QSize(1,1))
        self.bbox.setText(2, 'Marginal 2D')
        self.bbox.Connect(2, self.PlotMarg2D)
        self.bbox.addButton()
        self.bbox.setIconSize(3, QtCore.QSize(1,1))
        self.bbox.setText(3, 'Save')
        self.bbox.Connect(3, self.Archive)
    self.area.resize(self.defxy[0], self.defxy[1])
    self.form.resize(self.defxy[0], self.defxy[1])
    if self.stem is not None:
      try:
        wid = self.form.Widget
      except:
        wid = self.form
      wid.setWindowTitle(self.stem)
    return self.form
  def clrGUI(self):
    self.gbox0.remove(self.grap)
    self.dock0.remove()
    self.dock1.remove()
    self.dock2.remove()
    del self.dock0
    del self.dock1
    del self.dock2
    del self.area
  def close(self):
    if self.form is None: return
    self.form.close()
    self.form = None
  def SetBins(self, ev = None):
    bw = '1' if self.defbw is None else str(self.defbw)
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.EDx = lbw.LBWidget(self.Dlg, "Enter bin width: ", 1, 'edit', bw)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.SetBinsHL)
    self.BBx.Widgets[1].connect("btndown", self.SetBinsOK)
    self.Box.add(self.EDx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def SetBinsHL(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[4]+MANSUFF)
  def SetBinsOK(self, ev = None):
    if not(self.EDx.valiData(1, 1)): return
    self.uibw = self.EDx.retData()
    self.uibw = None if not(isnumeric(self.uibw)) else float(self.uibw)
    self.Dlg.close()
    del self.Dlg
    self.PlotHist(ev, self.uibw)
  def SetRes(self, ev = None):
    self.imv = [[]] * self.NX
    for i in range(self.NX):
      self.imv[i] = "".join( (self.labels[i], ": Mean=", str(self.mn[i]), "; Var.=", str(self.vr[i])) )
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBi = lbw.LBWidget(self.Dlg, "Data selection", 1,"listbox", None, self.imv)
    self.LBi.setMode(3, range(self.NX))
    self.EDs = lbw.LBWidget(self.Dlg, 'Probability of release sampling resolution: ', 1, 'edit', '128')
    self.EDv = lbw.LBWidget(self.Dlg, 'Coefficient of varation sampling resolution: ', 1, 'edit', '64')
    self.EDa = lbw.LBWidget(self.Dlg, "Heterogeneity sampling resolution (leave as `1' for homogeneous model): ", 1, 'edit', '1')
    self.EDn = lbw.LBWidget(self.Dlg, "Number of release sites sampling resolution: ", 1, 'edit', '16')
    self.EDN = lbw.LBWidget(self.Dlg, "Max. no. release sites (ideally integer*(n_res-1)+1)", 1, 'edit', '16')
    self.RBx = lbw.LBWidget(self.Dlg, 'Reciprocal prior for number of release sites', 0, 'radiobutton', True)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button", "Button"], ["Help", "Cancel", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.SetResHL)
    self.BBx.Widgets[1].connect("btndown", self.SetResCC)
    self.BBx.Widgets[2].connect("btndown", self.SetResOK)
    self.Box.add(self.LBi)
    self.Box.add(self.EDs)
    self.Box.add(self.EDv)
    self.Box.add(self.EDa)
    self.Box.add(self.EDn)
    self.Box.add(self.EDN)
    self.Box.add(self.RBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def SetResHL(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[2]+MANSUFF)
  def SetResCC(self, ev = None):
    self.Dlg.close()
    del self.Dlg
  def SetResOK(self, ev = None):
    uimv = self.LBi.retData()
    OK = True
    OK = OK and len(uimv)
    OK = OK and self.EDs.valiData(1, 1)
    OK = OK and self.EDv.valiData(1, 1)
    OK = OK and self.EDa.valiData(1, 1)
    OK = OK and self.EDn.valiData(1, 1)
    OK = OK and self.EDN.valiData(1, 1)
    if not(OK): return
    sres = abs(int(self.EDs.retData()))
    vres = abs(int(self.EDv.retData()))
    ares = abs(int(self.EDa.retData()))
    nres = abs(int(self.EDn.retData()))
    nmax = abs(int(self.EDN.retData()))
    ngeo = int(self.RBx.retData())
    self.Dlg.close()
    del self.Dlg
    _X = [[]] * len(uimv)
    for i in range(len(uimv)):
      _X[i] = nanravel(self.X[uimv[i]])
    self.setData(_X, self.e)
    self.setRes(nres, ares, vres, sres)
    self.setLimits(None, None, None, [1, nmax], ngeo)
    _pgb = pgb()
    self.setPriors(False, _pgb)
    self.calcPosts(USEPARALLEL, _pgb)
    _pgb.close()
    self.clrGUI()
    self.iniGUI().show()
  def RunDGEI(self, ev = None):
    self.imv = [[]] * self.NX
    for i in range(self.NX):
      self.imv[i] = "".join( (self.labels[i], ": Mean=", str(self.mn[i]), "; Var.=", str(self.vr[i])) )
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBi = lbw.LBWidget(self.Dlg, "Data selection", 1,"listbox", None, self.imv)
    self.LBi.setMode(3, range(self.NX))
    self.EDq = lbw.LBWidget(self.Dlg, 'Quantal size sampling resolution: ', 1, 'edit', '128')
    self.EDg = lbw.LBWidget(self.Dlg, 'Gamma shape parameter sampling resolution: ', 1, 'edit', '64')
    self.EDn = lbw.LBWidget(self.Dlg, "Number of release sites sampling resolution: ", 1, 'edit', '16')
    self.EDN = lbw.LBWidget(self.Dlg, "Max. no. release sites (ideally integer*(n_res-1)+1)", 1, 'edit', '16')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button", "Button"], ["Help", "Cancel", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.RunDGEIHL)
    self.BBx.Widgets[1].connect("btndown", self.RunDGEICC)
    self.BBx.Widgets[2].connect("btndown", self.RunDGEIOK)
    self.Box.add(self.LBi)
    self.Box.add(self.EDq)
    self.Box.add(self.EDg)
    self.Box.add(self.EDn)
    self.Box.add(self.EDN)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def RunDGEIHL(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[2]+MANSUFF)
  def RunDGEICC(self, ev = None):
    self.Dlg.close()
    del self.Dlg
  def RunDGEIOK(self, ev = None):
    uimv = self.LBi.retData()
    OK = True
    OK = OK and len(uimv)
    OK = OK and self.EDq.valiData(1, 1)
    OK = OK and self.EDg.valiData(1, 1)
    OK = OK and self.EDn.valiData(1, 1)
    OK = OK and self.EDN.valiData(1, 1)
    if not(OK): return
    qres = abs(int(self.EDq.retData()))
    gres = abs(int(self.EDg.retData()))
    nres = abs(int(self.EDn.retData()))
    nmax = abs(int(self.EDN.retData()))
    self.Dlg.close()
    del self.Dlg
    _X = [[]] * len(uimv)
    for i in range(len(uimv)):
      _X[i] = nanravel(self.X[uimv[i]])
    self.dgei = BQA([1, nmax])
    self.dgei.set_data(_X, self.e)
    _pgb = pgb()
    self.nres = nres
    self.qres = qres
    self.gres = gres
    self.dgei.dgei(nres, qres, gres, pgb=_pgb)
    _pgb.close()
    self.hatn = self.dgei.hatn
    self.hatq = self.dgei.hatq * self.pol
    self.hatg = self.dgei.hatg
    self.hatv = self.dgei.hatv
    self.hatl = self.dgei.hatl
    self.hate = self.dgei.hate
    self.hata = np.NaN
    self.hats = self.dgei.hatp
    self.hatr = self.hatn * self.hatq
    self.hatvr = QLV(np.abs(self.mn), self.hate, self.hatn, self.hatg, abs(self.hatl), self.hata)
    self.PN = discprob.mass(self.dgei.marg_n.prob, [np.ravel(self.dgei.marg_n.vals['n'])])
    self.PQ = discprob.mass(self.dgei.marg_q.prob, [np.ravel(self.dgei.marg_q.vals['q'])])
    self.PG = discprob.mass(self.dgei.marg_g.prob, [np.ravel(self.dgei.marg_g.vals['g'])])
    self.PV = discprob.mass(self.dgei.marg_g.prob, [1./np.sqrt(np.ravel(self.dgei.marg_g.vals['g']))])
    self.clrGUI()
    self.iniGUI().show()
  def PlotHist(self, ev = None, bw = None):
    self.out.append(self.plotHist(None, bw))
    self.out[-1].resize(self.defxy[0], self.defxy[1])
    self.out[-1].show()
  def PlotMarg1D(self, ev = None):
    self.out.append(self.plotMarg1D())
    self.out[-1].resize(self.defxy[0], self.defxy[1])
    self.out[-1].show()
  def PlotMarg2D(self, ev = None):
    self.cbt = ['P(R, Q)', 'P(Q, gamma)', 'P(R, gamma)', 'P(gamma, alpha)']
    if not(self.modelBeta): self.cbt = self.cbt[:3]
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Parameters: ', 1, 'combobox', 0, self.cbt)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Mass function', 0, ["Joint", "Conditional"], 'radiobutton', 0)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.SetBinsHL)
    self.BBx.Widgets[1].connect("btndown", self.PlotMarg2DOK)
    self.Box.add(self.CBx)
    self.Box.add(self.BGB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def PlotMarg2DOK(self, ev = None):
    spec = self.CBx.retData()
    cond = self.BGB.retData()
    self.Dlg.close()
    self.out.append(self.plotMarg2D(spec, cond))
    self.out[-1].resize(self.defxy[0], self.defxy[1])
    self.out[-1].show()
  def Export(self, _T, s = '_'):
    defop = 'results' + s + '.tab'
    if self.path is not None or self.stem is not None:
      defop = self.path + '/' + self.stem + s + '.tab'
    _base = lbw.LBWidget(None, None, None, 'base')
    _opfn = self.Base.dlgFileSave("Export File", defop, "Results file (*.tab);;All (*.*)")
    if _opfn is not None:
      if len(_opfn):
        writeDTFile(_opfn, _T)
  def ExpHist(self, ev = None):
    self.lbt = ['Histograms', 'Projected QLF', 'Marginal release probabilities']
    if np.isnan(self.hatn): self.lbt = self.lbt[:1]
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBx = lbw.LBWidget(self.Dlg, "Export selection", 1,"listbox", None, self.lbt)
    self.LBx.setMode(3, range(len(self.lbt)))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.SetBinsHL)
    self.BBx.Widgets[1].connect("btndown", self.ExpHistOK)
    self.Box.add(self.LBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def ExpHistOK(self, ev = None):
    uix = self.LBx.retData()[:]
    self.Dlg.close()
    if not(len(uix)): return
    Ylbl = self.labels
    HS = []
    if 0 in uix:
      HS.append(xY2list(self.xHist[0], self.yHist, 'Amplitude', Ylbl))
    if 1 in uix:
      HS.append(xY2list(self.xQLF[0], self.yQLF, 'Amplitude', Ylbl))
    if 2 in uix:
      HS.append(xY2list(self.s, self.PMS.P, 'Probability', Ylbl))
    self.Export(HS, '_hs')
  def ExpMarg1D(self, ev = None):
    self.lbt = ['r', 'q', 'gamma', 'alpha'] if self.modelBeta else ['r', 'q', 'gamma']
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LB0 = lbw.LBWidget(self.Dlg, "Marginal selection", 1,"listbox", None, self.lbt)
    self.LB0.setMode(3, range(len(self.lbt)))
    self.LB1 = lbw.LBWidget(self.Dlg, "Derived marginal selection", 1,"listbox", None, ['CV', 'n'])
    self.LB1.setMode(3, range(2))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"])
    self.BBx.Widgets[0].connect("btndown", self.SetBinsHL)
    self.BBx.Widgets[1].connect("btndown", self.ExpMarg1DOK)
    self.Box.add(self.LB0)
    self.Box.add(self.LB1)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def ExpMarg1DOK(self, ev = None):
    ui0 = self.LB0.retData()[:]
    ui1 = self.LB1.retData()[:]
    self.Dlg.close()
    if not(len(ui0)) and not(len(ui1)): return
    lt = []
    lb0 = [['r', 'P(r)'], ['q', 'P(q)'], ['gamma', 'P(gamma)'], ['alpha', 'P(alpha)']]
    lb1 = [['CV', 'P(CV)'], ['n', 'P(n)']]
    dt0 = [[self.PR.X[0], self.PR.P], [self.PQ.X[0], self.PQ.P], [self.PG.X[0], self.PG.P], [self.PA.X[0], self.PA.P]]
    dt1 = [[self.PV.X[0], self.PV.P], [self.PN.X[0], self.PN.P]]
    M1 = []
    for i in range(4):
      if i in ui0:
        M1.append(xy2list(dt0[i][0], dt0[i][1], lb0[i][0], lb0[i][1]))
    for i in range(3):
      if i in ui1:
        M1.append(xy2list(dt1[i][0], dt1[i][1], lb1[i][0], lb1[i][1]))
    self.Export(M1, '_m1')
  def ExpMarg2D(self, ev = None):
    if self.P3d.NP < 3:
      self.Export([self.P3d.X, self.P3d.P], '_m2')
      return
    _X = [[]] * (self.mres+1)
    _X[0] = self.P3d.X
    for i in range(self.mres):
      _X[i+1] = self.P3d.P[i]
    self.Export(_X, '_m2')

