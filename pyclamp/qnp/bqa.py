# This module is a slow Qt/matplotlib-based implementation of BQA.
# BUT! It is obsolete, untested and might not work: use qmod.py instead.

import sys
import os
import webbrowser
import matplotlib as mpl
from iplot import *
import numpy as np
import scipy as sp
import scipy.stats as stats
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib as mpl
import matplotlib.pyplot as mp
import matplotlib.ticker as mt
import matplotlib.pyplot as mp
from fpfunc import *
import lbwgui as lbw
from iofunc import *
from fpfunc import *
from dtypes import *
from lsfunc import *
from fpfunc import *
from iofunc import *
import discprob
import time

CWDIR = os.getcwd()
MANDIR = CWDIR + '/../pyclamp/man/'
MANPREF = "pyclamp"
MANSUFF = ".html"
MANHELP = ["su12", "su13", "su14", "su15", "su16", "su17"]

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

class qbay:
  def __init__ (self, rescoef = 1.0, resmin = 128, resmax = 192):
    self.maxn = 0
    self.NX = 0
    self.res = 0
    self.nlims = None
    self.modelBeta = True
    self.iniHatValues()
    self.setRes(rescoef, resmin, resmax)     
  def iniHatValues(self):    
    self.hate = np.NaN
    self.hatq = np.NaN
    self.hatg = np.NaN
    self.hata = np.NaN
    self.hatr = np.NaN
    self.hatn = np.NaN
    self.hatl = np.NaN
    self.hatv = np.NaN
    self.hats = np.NaN
  def setRes(self, rescoef = None, resmin = None, resmax = None):
    if isarray(rescoef):
      if len(rescoef) == 2:
        self.sres = rescoef[0]
        self.vres = rescoef[1]
        self.ares = 1
        self.alims = [1e6, 1e6]
      elif len(rescoef) == 3:
        self.sres = rescoef[0]
        self.vres = rescoef[1]
        self.ares = rescoef[2]
      self.qres = max(self.sres, self.vres)
      self.gres = self.vres
      self.rres = self.qres
      if self.nlims is None: return
      self.nres = int(self.nlims[1] - self.nlims[0] + 1)
      self.ires = int(self.nlims[1]+1)
      self.mres = self.NX
      return
    if not(rescoef is None): self.resCoef = rescoef    
    if not(resmin is None): self.resMin = resmin
    if not(resmax is None): self.resMax = resmax
    if not(self.maxn) or not(self.NX) or self.nlims is None: return
    self.res = max(self.resMin, min(self.resMax, self.resCoef * self.maxn))    
    self.sres = max(self.res, 3)
    self.mres = self.NX
    self.nres = int(self.nlims[1] - self.nlims[0] + 1)
    self.ires = int(self.nlims[1]+1)
    if not(isarray(rescoef)):
      if  self.modelBeta:
        self.ares = max(int(np.sqrt(self.res)), 3) 
        self.vres = max(int(self.res/self.ares), 3)    
      else:  
        self.ares = 1
        self.vres = max(self.res, 3)      
        self.alims = [1e6, 1e6]
    self.qres = max(self.res, 3)  
    self.gres = self.vres
    self.rres = max(self.res, 3)
  def setData(self, _X, _e = None, _nmax = None):
    if type(_X) is list:
      self.X = list2nan2D(_X)
    else:
      self.X = _X
    self.Xx = nanravel(self.X)
    self.nx = nancnt(self.X, axis = 1)
    self.NX = len(self.nx)
    self.Nx = np.sum(self.nx)
    self.mn = nanmean(self.X, axis = 1)
    self.vr = nanvar(self.X, axis = 1)    
    self.vrvr = nanvrvr(self.X, axis = 1)
    self.maxn = np.max(nancnt(self.X, axis = 1))
    self.sd = np.sqrt(self.vr)
    self.cd = self.vr / self.mn
    self.cv = self.sd / self.mn
    self.minx = np.min(self.Xx)
    self.maxx = np.max(self.Xx)    
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
    self.setNoise(_e)
    self.setLimits(None, None, None, None, _nmax)
  def setNoise(self, _e = None): # if an array of two, it defines a range
    mino = 1e-300
    if _e is None: _e = [0., np.inf]
    if type(_e) is list:
      if len(_e) == 2:
        y = self.Xx[self.Xx <= 0] if self.pol > 0 else self.Xx[self.Xx >= 0]
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
  def fitParabola(self, alpha = 0.05):
    sdc = 3.0
    mino = 1e-300
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
  def setLimits(self, _slims = None, _vlims = None, _alims = None, _nlims = None, _nmax = None):
    if _slims is None:
      _slims = [0.04, 0.96]
    if _vlims is None:  
      _vlims = [0.05, 1.0]
    if _alims is None:
      _alims = [0.2, 20.0]      
    if _nlims is None:
      rhat = self.amxm
      rmin = self.amnm
      rmax = rhat * 2.0
      qhat = self.acdm
      _nlims = [np.floor(rmin/qhat), np.ceil(rmax/qhat)]
      _nlims[0] = min(_nlims[0], 1)
      _nlims[0] += not(_nlims[0])
      _nlims[1] = max(_nlims[1], _nlims[0]+1)
      if not(_nmax is None):
        _nlims[1] = min(_nlims[1], _nmax)
    self.slims = _slims  
    self.vlims = _vlims
    self.alims = _alims
    self.nlims = _nlims
    self.ilims = [0, self.nlims[1]]
  def setPriors(self, rescoef = None, resmin = None, resmax = None, _modelBeta = True):
    self.modelBeta = _modelBeta
    self.setRes(rescoef, resmin, resmax)    
    self.Pr = discprob.mass()    
    self.Pr.setx([self.sres, self.vres, self.ares, self.nres, self.ires, self.mres], [-2, 1, 1, 0, 0, 0], self.slims, self.vlims, self.alims, self.nlims, self.ilims, self.mn)
    self.Pr.setUniPrior() # [0, 0, 0, 2, 0, 0]
    self.s = self.Pr.X[0]
    self.v = self.Pr.X[1]
    self.a = self.Pr.X[2]
    self.n = self.Pr.X[3]                 
    self.m = self.Pr.X[5]
    self.S = self.Pr.repa(0)        
    self.V = self.Pr.repa(1)
    self.A = self.Pr.repa(2)
    self.N = self.Pr.repa(3)
    self.I = self.Pr.repa(4)    
    self.M = self.Pr.repa(5)    
    self.B = self.A/self.S - self.A
    if not(self.modelBeta):
      #self.C = stats.binom.pmf(self.I, self.N, self.S)  risks memory errors (blame scipy.stats)
      self.C = np.zeros((self.sres, self.vres, self.ares, self.nres, self.ires, self.mres), dtype = float)
      for i in range(self.mres): 
        self.C[:, :, :, :, :, i] = stats.binom.pmf(self.I[:, :, :, :, :, i], self.N[:, :, :, :, :, i], self.S[:, :, :, :, :, i])      
    else: 
      # --- the short but sadly incorrect way ---
    
      # self.C = discprob.betabinopmf(self.I, self.N, self.A, self.B) 
    
      # --- the long way ----
      #'''
      pb = pbfig("Constructing priors")
      pb.setup(['$n$', r'$\mu$'], [self.nres, self.sres*self.ares], ['r', 'b'])    
      sr = np.tile(self.s.reshape(self.sres, 1), (1, self.ares))
      ar = np.tile(self.a.reshape(1, self.ares), (self.sres, 1))
      br = ar/sr - ar
      self.C = np.zeros((self.sres, self.vres, self.ares, self.nres, self.ires, self.mres), dtype = float)
      for i in range(self.nres):
        h = self.n[i]+1
        pb.forceupdate([i, 0])
        bp = discprob.betanomial(self.n[i], ar, br, pb)
        self.C[:, :, :, i, :h, :] = np.tile(bp.reshape(self.sres, 1, self.ares, h, 1), (1, self.vres, 1, 1, self.mres))  
      pb.close()  
      #'''
      
      ''' 
      # --- the very long way ---
      self.C = np.zeros((self.sres, self.vres, self.ares, self.nres, self.ires, self.mres), dtype = float)
      lh = discprob.mass()
      pr = discprob.mass()
      pb = pbfig("Constructing priors")
      pb.setup(['$n$', '$s$', r'$\alpha$'], [self.nres, self.sres, self.ares], ['r', 'g', 'b'])    
      for i in range(self.nres):
        ni = int(self.n[i])
        h = ni+1
        lh.setP()
        lh.setX([np.array((0, 1), dtype = int)]*ni)
        pr.setP()
        pr.setx([h], [0], np.arange(h, dtype = int))      
        H = np.zeros(lh.nX, dtype = int)
        for l in range(ni):
          H += lh.repa(l)        
          for j in range(self.sres):
            sj = self.s[j]
            for k in range(self.ares):
              pb.update([i, j, k])
              sk = discprob.betasample(ni, self.a[k], None, sj)
              mpr = [[]] * ni
              for l in range(ni):
                mpr[l] = np.array( (1. - sk[l], sk[l]) )
              lh.setp(mpr)
            pr.setUniPrior()
            pr.calcPost(lh, H)
            self.C[j, :, k, i, :h, :] = np.tile(pr.P.reshape(1, h, 1), (self.vres, 1, self.mres))
      pb.close()      
      '''
    self.Q = self.M / ( self.N * self.S)
    self.G = 1.0 / (self.V ** 2.0)
    self.L = np.fabs(self.Q) / self.G    
    self.GI = self.G * self.I    
    self.g = 1.0 / (self.v ** 2.0)
    self.g.sort()
    self.qlims = [self.pol*np.fabs(self.Q).min(), self.pol*np.fabs(self.Q).max()]
    self.rlims = [self.qlims[0]*self.nlims[0], self.qlims[1]*self.nlims[1]]
    self.Pr.normalise()
  def calcPosts(self):
    mino = 1e-300    
    self.P = self.Pr.marginalise(4)
    pb = pbfig("Constructing posteriors")
    self.LL = [[]] * self.mres
    self.LLmax = np.empty(self.mres, dtype = float)
    for i in range(self.mres):
      Xi = self.pol * nanravel(self.X[i])
      pb.setup(['Data set', 'Datum'], [self.mres, len(Xi)], ['r', 'b'])      
      pb.update([i, 0])
      pr = discprob.mass(self.Pr.P.sum(4)[:, :, :, :, i], self.Pr.X[:4])
      self.LL[i] = np.log(pr.P + mino)
      pb.setlast("Calculating...", len(Xi), 'b')
      for j in range(len(Xi)):
        pb.updatelast(j)
        Pj = self.C[:, :, :, :, 0, i] * stats.norm.pdf(Xi[j], 0.0, self.e)
        if Xi[j] >= 0:
          #Gj = self.C[:, :, :, :, 1:, i] * stats.gamma.pdf(Xi[j], self.AI[:, :, :, :, 1:, i], 0.0, self.B[:, :, :, :, 1:, i]) - buggy (blame scipy.stats)
          '''
          Gj = self.C[:, :, :, :, 1:, i] * discprob.gampdf(Xi[j], self.GI[:, :, :, :, 1:, i], self.L[:, :, :, :, 1:, i])                    
          Pj += np.sum(Gj, 4)
          '''
          for k in range(self.nres):
            ni = self.n[k] + 1
            Gj = self.C[:, :, :, k, 1:ni, i] * discprob.gampdf(Xi[j], self.GI[:, :, :, k, 1:ni, i], self.L[:, :, :, k, 1:ni, i])
            Pj[:, :, :, k] += np.sum(Gj, 3)
        
        self.LL[i] += np.log(Pj + mino)
      self.LLmax[i] = self.LL[i].max()
      pr.setP(np.exp(self.LL[i] - self.LLmax[i]))
      pr.normalise()
      self.P.P[:, :, :, :, i] = pr.P.copy(); # * float(self.nx[i])/float(self.Nx)
    pb.close()
    self.PSVANM = self.P.copy()    
    self.calcCondPosts()
    self.calcMargPosts()            
    self.calcHatValues()    
  def calcCondPosts(self):
    mino = 1e-300
    self.PQGARM = discprob.mass()
    self.PQGARM.setx([self.qres, self.gres, self.ares, self.rres, self.mres], [1, 0, 1, 1, 0], self.qlims, self.g, self.a, self.rlims, self.m)
    self.PQGARM.setUniPrior()
    self.q = self.PQGARM.X[0]            
    self.g = self.PQGARM.X[1]
    self.a = self.PQGARM.X[2]
    self.r = self.PQGARM.X[3]
    pb = pbfig("Constructing conditional posteriors...")
    pb.setup('Data set', self.mres, 'r') 
    for i in range(self.mres):      
      pb.update(i)
      pr = discprob.mass(self.PQGARM.P[:, :, :, :, i], self.PQGARM.X[:4])
      lh = discprob.mass(self.PSVANM.P[:, :, :, :, i], self.PSVANM.X[:4])      
      lhR = self.m[i] / (lh.repa(0)+mino)
      lhG = 1.0 / (lh.repa(1)**2.0 + mino)
      lhA = lh.repa(2)
      lhQ = lhR / (lh.repa(3)+mino)      
      pr.calcPost(lh, lhQ, lhG, lhA, lhR)
      pr.normalise();
      self.PQGARM.P[:, :, :, :, i] = pr.P.copy(); # * float(self.nx[i])/float(self.Nx)                 
    pb.close() 
    _PQGARM  = self.PQGARM.copy()
    inc = 1e-15
    done = False
    while not(done): # horrible hack to prevent crashes from highly discordant data
      self.PQGAR = _PQGARM.multiply(4)
      if self.PQGAR.P.max() > mino:
        done = True
      else:
        _PQGARM.P += inc
    self.PQGAR.normalise()  
    self.absq = np.fabs(self.q)   
    self.absg = np.fabs(self.g)
    self.absa = np.fabs(self.a)
    self.absr = np.fabs(self.r)    
    self.absn = np.fabs(self.n)
    pr = discprob.mass()
    pr.setX(self.n)
    pr.setUniPrior()
    lh = self.PQGAR.copy()
    lhN = np.fabs(lh.repa(3) / lh.repa(0)+mino)
    pr.calcPost(lh, lhN)
    pr.normalise()
    self.PN = pr.copy()
  def calcMargPosts(self):    
    self.PQGAM = self.PQGARM.marginalise(3)
    self.PQGRM = self.PQGARM.marginalise(2)      
    self.PQRAM = self.PQGARM.marginalise(1)
    self.PGRAM = self.PQGARM.marginalise(0)
    self.PQAM = self.PQGAM.marginalise(1)
    self.PGAM = self.PQGAM.marginalise(0)
    self.PQGM = self.PQGRM.marginalise(2)    
    self.PQRM = self.PQGRM.marginalise(1)
    self.PGRM = self.PQGRM.marginalise(0)
    self.PQGA = self.PQGAR.marginalise(3)
    self.PQGR = self.PQGAR.marginalise(2)
    self.PQAR = self.PQGAR.marginalise(1)
    self.PGAR = self.PQGAR.marginalise(0)
    self.PQA = self.PQAR.marginalise(2)  
    self.PGA = self.PGAR.marginalise(2)
    self.PAR = self.PGAR.marginalise(0)
    self.PQG = self.PQGR.marginalise(2)
    self.PQR = self.PQGR.marginalise(1)
    self.PGR = self.PQGR.marginalise(0)
    self.PQ = self.PQG.marginal(0)
    self.PG = self.PQG.marginal(1)   
    self.PA = self.PAR.marginal(0)
    self.PR = self.PQR.marginal(1)
    self.PSM = discprob.mass()
    self.PSM.setx([self.sres, self.mres], [0, 0], self.s, self.m)
    self.PSM.setUniPrior()
    for i in range(self.mres):
      pr = discprob.mass(self.PSM.P[:, i], self.PSM.X[0])
      pr.calcPost(self.PR, np.fabs(self.m[i] / self.PR.X[0]))
      self.PSM.P[:, i] = pr.P / sum(pr.P)    
  def calcHatValues(self):    
    self.hatq = self.PQ.sample(1)
    self.hatg = self.PG.sample(1)
    self.hata = self.PA.sample(1)
    self.hatr = self.PR.sample(1)
    self.hatn = self.hatr / self.hatq    
    self.hatl = self.hatq / self.hatg
    self.hatv = 1.0 / np.sqrt(self.hatg)
    #self.hatv = np.sqrt((self.hatv * self.hatq) ** 2.0 - self.hate**2.0)/self.hatq - we now neglect the effect of self.e
    self.hats = np.empty(self.mres, dtype = float)
    for i in range(self.mres):
      pr = discprob.mass(self.PSM.P[:,i], self.s)
      self.hats[i] = self.m[i]/self.hatr # pr.sample(1)
    self.hatvr= QLV(self.mn, self.hate, self.hatn, self.hatg, self.hatl, self.hata)  
  def plotMarg0(self, fi = None, _nr = None, _nc = None, counter = None): # plots direct marginals
    if fi is None: fi = mp.figure()
    nr = 2
    nc = 4 if self.modelBeta else 3
    if _nr is not None: nr = _nr
    if _nc is not None: nc = _nc
    if counter is None: counter = 0
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absq, self.PQ.P)
    mp.xlabel('$q$ / units');
    mp.ylabel('Posterior / probability mass');
    mp.xscale('log')
    mp.gca().xaxis.set_major_formatter(mt.ScalarFormatter())
    mp.axis('tight')
    mp.title(r"".join(("$\hat{q}=", str(float(self.hatq)), "$")))
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absg, self.PG.P)
    mp.xlabel(r'$\gamma$ / normalised units');
    mp.ylabel('Posterior / probability mass'); 
    mp.xscale('log')
    mp.gca().xaxis.set_major_formatter(mt.ScalarFormatter())
    mp.axis('tight')
    mp.title(r"".join(("$\hat{\gamma}=", str(float(self.hatg)), "$")))
    counter += 1  
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absr, self.PR.P)
    mp.xlabel('$r$ / units');
    mp.ylabel('Posterior / probability mass');        
    mp.xscale('log')
    mp.gca().xaxis.set_major_formatter(mt.ScalarFormatter())
    mp.axis('tight')
    mp.title(r"".join(("$\hat{r}=", str(float(self.hatr)), "$")))
    if self.modelBeta:
      counter += 1
      fi.add_subplot(nr,nc,counter)
      mp.plot(self.absa, self.PA.P)
      mp.xlabel(r'$\alpha$ / normalised units');
      mp.ylabel('Posterior / probability mass'); 
      mp.xscale('log')
      mp.gca().xaxis.set_major_formatter(mt.ScalarFormatter())
      mp.axis('tight')
      mp.title(r"".join(("$\hat{a}=", str(float(self.hata)), "$")))
    return fi
  def plotMarg1(self, fi = None, _nr = None, _nc = None, counter = None): # plots derived marginals
    if fi is None: fi = mp.figure()
    nr = 2
    nc = 3
    if _nr is not None: nr = _nr
    if _nc is not None: nc = _nc
    if counter is None: counter = nc
    counter += 1  
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absn+0.5, self.PN.P, linestyle='steps')
    mp.xlabel('$n$ / sites');
    mp.ylabel('Posterior / probability mass');        
    mp.axis('tight')
    mp.title(r"".join(("$\hat{n}=", str(float(self.hatn)), "$")))
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.v, self.PG.P)
    mp.xlabel(r'$v$ / normalised units');
    mp.ylabel('Posterior / probability mass'); 
    mp.xscale('log')
    mp.gca().xaxis.set_major_formatter(mt.ScalarFormatter())
    mp.axis('tight')
    mp.title(r"".join(("$\hat{v}=", str(float(self.hatv)), "$")))
    return fi
  def plotMarg2(self, fi = None, _nr = None, _nc = None, counter = None):
    if fi is None: fi = mp.figure()
    nr = 2
    nc = 2
    if _nr is not None: nr = _nr
    if _nc is not None: nc = _nc
    if counter is None: counter = 0
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.pcolor(self.absr, self.absq, self.PQR.P, antialiased = False)
    mp.colorbar()
    mp.xscale('log')
    mp.yscale('log')
    mp.xlabel('$r$ / units')
    mp.ylabel('$q$ / units') 
    mp.axis('tight')
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.pcolor(self.absg, self.absq, self.PQG.P, antialiased = False)
    mp.colorbar()
    mp.xscale('log')
    mp.yscale('log')
    mp.xlabel(r'$\gamma$ / normalised units');
    mp.ylabel('$q$ / units');
    mp.axis('tight')
    self.PGR = self.PQGR.marginalise(0)
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.pcolor(self.absr, self.absg, self.PGR.P, antialiased = False)
    mp.colorbar()
    mp.xscale('log')
    mp.yscale('log')
    mp.xlabel(r'$r$ / units');
    mp.ylabel(r'$\gamma$ / normalised units');
    mp.axis('tight')  
    if self.modelBeta:
      counter += 1
      fi.add_subplot(nr,nc,counter)
      mp.pcolor(self.absa, self.absg, self.PGA.P, antialiased = False)
      mp.colorbar()
      mp.xscale('log')
      mp.yscale('log')
      mp.xlabel(r'$\alpha$ / normalised units');
      mp.ylabel(r'$\gamma$ / normalised units');
      mp.axis('tight')  
    return fi
  def summTable(self, fi = None, numformat = "%.3f"):
    if fi is None: fi = mp.figure()
    hata = np.NaN
    if self.modelBeta:      
      hata = self.hata
    self.results = [] 
    self.results.append([r'$N$', str(int(self.nx.min())) + "-" + str(int(self.nx.max()))])
    self.results.append([r'$\vert \mu \vert$', str(int(self.NX))])    
    self.results.append([r'$<\epsilon>$', numformat % float(self.hate)])
    self.results.append([r'$<Q>$', numformat % float(self.parq)])
    self.results.append([r'$<N>$', numformat % float(self.parn)])
    self.results.append(['Resn', str(int(self.sres))])
    self.results.append([r'$\hat{q}$', numformat % float(self.hatq)])
    self.results.append([r'$\hat{\gamma}$', numformat % float(self.hatg)])
    self.results.append([r'$\hat{\alpha}$', numformat % float(hata)])
    self.results.append([r'$\hat{r}$', numformat % float(self.hatr)])
    self.results.append([r'$\hat{\lambda}$', numformat % float(self.hatl)])  
    self.results.append([r'$\hat{v}$', numformat % float(self.hatv)])
    self.results.append([r'$\hat{q}\hat{v}$', numformat % float(self.hatq*self.hatv)])
    self.results.append([r'$\hat{n}$', numformat % float(self.hatn)])
    itp = itxtplot()
    itp.setData(self.results, None, 12)
    return itp
  def plotSummary(self, fi = None): # A GPU-undemanding summary plot
    if fi is None: fi = mp.figure()
    self.plotMarg0(fi, 2, 4)
    self.plotMarg1(fi, 2, 4)
    fi.add_subplot(2, 4, 7)
    self.plotMoments()
    mp.title(r"".join(("$\epsilon=", str(self.hate), "$")))
    fi.canvas.draw()
    fi.add_subplot(2, 4, 8)
    self.summTable(fi)
  def plotMargPosts(self, _nr = None, _nc = None):
    counter = 0    
    nr = 2
    nc = 4 if self.modelBeta else 3
    if _nr is not None: nr = _nr
    if _nc is not None: nc = _nc
    counter = 0      
    fi = mp.figure()
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absq, self.PQ.P)
    mp.xlabel('$q$ / number');
    mp.ylabel('Posterior / probability mass');
    mp.xscale('log')
    mp.axis('tight')
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absg, self.PG.P)
    mp.xlabel(r'$\gamma$ / normalised units');
    mp.ylabel('Posterior / probability mass'); 
    mp.xscale('log')
    mp.axis('tight')
    if self.modelBeta:
      counter += 1
      fi.add_subplot(nr,nc,counter)
      mp.plot(self.absa, self.PA.P)
      mp.xlabel(r'$\alpha$ / normalised units');
      mp.ylabel('Posterior / probability mass'); 
      mp.xscale('log')
      mp.axis('tight')
    counter += 1  
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absr, self.PR.P)
    mp.xlabel('$r$ / units');
    mp.ylabel('Posterior / probability mass');        
    mp.xscale('log')
    mp.axis('tight')
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.plot(self.absn+0.5, self.PN.P, linestyle='steps')
    mp.xlabel('$n$ / sites');
    mp.ylabel('Posterior / probability mass');        
    mp.axis('tight')
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.pcolor(self.absr, self.absq, self.PQR.P, antialiased = False)
    mp.xscale('log')
    mp.yscale('log')
    mp.xlabel('$r$ / units')
    mp.ylabel('$q$ / units') 
    mp.axis('tight')
    counter += 1
    fi.add_subplot(nr,nc,counter)
    mp.pcolor(self.absg, self.absq, self.PQG.P, antialiased = False)
    mp.xscale('log')
    mp.yscale('log')
    mp.xlabel(r'$\gamma$ / normalised units');
    mp.ylabel('$q$ / units');
    mp.axis('tight')
    if self.modelBeta:
      counter += 1
      fi.add_subplot(nr,nc,counter)
      mp.pcolor(self.absa, self.absg, self.PGA.P, antialiased = False)
      mp.xscale('log')
      mp.yscale('log')
      mp.xlabel(r'$\alpha$ / normalised units');
      mp.ylabel(r'$\gamma$ / normalised units');
      mp.axis('tight')  
    return fi
  def plotCondPosts(self, spec = 0): 
    nr = int(np.sqrt(self.mres))
    nc = int(np.ceil(float(self.mres) / float(nr) ))
    fi = mp.figure()
    if spec == 0:   
      for i in range(self.mres):
        fi.add_subplot(nr, nc, i+1)
        mp.plot(self.s, self.PSM.P[:,i])
        mp.xlabel('$s$ / per quantal site');
        mp.ylabel('Posterior / probability mass');    
        mp.axis('tight')        
    elif spec == 1:  
      for i in range(self.mres):
        fi.add_subplot(nr, nc, i+1)
        mp.pcolor(self.absr, self.absq, self.PQRM.P[:, :, i], antialiased = False)
        mp.xscale('log')
        mp.yscale('log')        
        mp.xlabel('$r$ / units');
        mp.ylabel('$q$ / units');        
        mp.axis('tight')    
    elif spec == 2:
      for i in range(self.mres):
        fi.add_subplot(nr, nc, i+1)
        mp.pcolor(self.absg, self.absq, self.PQGM.P[:, :, i], antialiased = False)
        mp.xscale('log')
        mp.yscale('log')
        mp.xlabel(r'$\gamma$ / normalised units');
        mp.ylabel('$q$ / units');        
        mp.axis('tight')          
    return fi  
  def plotCondHist(self):
    num = 1000
    fi = mp.figure()
    xh = np.linspace(self.minx, self.maxx, np.ceil(2.0*np.sqrt(self.maxn)))
    dxh = xh[1] - xh[0]
    hx = np.linspace(self.minx, self.maxx, num)    
    for i in range(self.NX):
      fi.add_subplot(2, self.NX, i+1)
      mp.plot(self.s, self.PSM.P[:,i])
      mp.xlabel('$s$ / per quantal site');
      mp.ylabel('Posterior / probability mass');    
      mp.axis('tight')
      fi.add_subplot(2, self.mres, self.mres+i+1)
      mp.hist(nanravel(self.X[i]), xh)        
      hy = QLF(hx, self.hate, np.round(self.hatn), self.hatg, self.hatl, self.hata, self.hats[i])
      hy *= self.nx[i] * dxh
      mp.plot(hx, hy.ravel(), 'r')
      mp.axis('tight')
      mp.xlabel('Response / units');
      mp.ylabel('Frequency / counts');      
  def overlayHist(self, bw = None, fi = None):
    if fi is None: fi = mp.figure()
    num = 1000
    nr = int(np.sqrt(self.mres))
    nc = int(np.ceil(float(self.mres) / float(nr) ))
    if bw is None: bw = self.defbw
    xh = binspace(self.minx, self.maxx, bw)
    dxh = xh[1] - xh[0]
    self.HXY = np.empty((self.mres+1, num), dtype = float) 
    hx = np.linspace(self.minx, self.maxx, num)
    self.HXY[0] = np.copy(hx)
    for i in range(self.mres):
      fi.add_subplot(nr, nc, i+1)
      mp.hist(nanravel(self.X[i]), xh)    
      hy = QLF(hx, self.hate, np.round(self.hatn), self.hatg, self.hatl, self.hata, self.hats[i])
      hy *= self.nx[i] * dxh
      self.HXY[i+1] = np.copy(hy)
      mp.plot(hx, hy.ravel(), 'r')
      mp.axis('tight')
      mp.xlabel('Response / units');
      mp.ylabel('Frequency / counts');      
      if i == 0:
        mp.title(''.join(('Bin width=', str(dxh))))
    return fi      
  def plotMoments(self):
    mino = 1e-300
    axc = 1.05;
    rxc = 0.99
    num = 1000;
    mp.errorbar(self.mn, self.vr, yerr = np.sqrt(self.vrvr), fmt = 'b.', ms = 12)
    mp.hold("true")
    maxr = -self.parq/self.paru if np.isnan(self.hatr) else self.hatr * float(round(self.hatn)) / self.hatn
    maxv = (self.vr+np.sqrt(self.vrvr)).max()
    if self.pol > 0:
      _hatmn = np.linspace(mino, float(maxr*rxc), num)
    else:
      _hatmn = np.linspace(float(self.pol*abs(maxr*rxc)), -mino, num)
    _hatvr = parafun(_hatmn, self.paru, self.parq, self.hate**2.0)
    mp.plot(_hatmn, _hatvr.ravel(), 'r')
    maxv = max(maxv, _hatvr.max())
    if not(np.isnan(self.hatn)):
      _hatvr = QLV(_hatmn, self.hate, self.hatn, self.hatg, self.hatl, self.hata)
      mp.plot(_hatmn, _hatvr.ravel(), 'g')
      maxv = max(maxv, _hatvr.max())  
    if self.pol > 0:
      mp.xlim([0, max(_hatmn[-1], self.mn.max())*axc])
    else:
      mp.xlim([min(_hatmn[0], self.mn.min())*axc, 0])      
    mp.ylim(0.0, axc * maxv)   
    mp.xlabel('$\mu$ / units');
    mp.ylabel('$\sigma^2$ / units$^2$');       

class bqa (qbay): # A Qt-free Matplotlib-API GUI wrapper class for qbay
  def __init__(self):
    qbay.__init__(self)
  def readFile(self, pf = "", spec = None): # an attempt of a universal import function
    #spec=None (or False) specifies row-arranged TAB data
    self.Labels = None
    if not(len(pf)): return
    ipdn, ipfn = os.path.split(pf)
    ipfs, ipfe = os.path.splitext(ipfn)
    ipfe = ipfe.lower()
    self.path = ipdn
    self.stem = ipfs
    # Handle easiest (or rather least-flexible) case - an Excel file
    if ipfe == ".xls" or ipfe == ".xlsx":
      self.Data = readXLData(pf) # this automatically tranposes and replaces strings with NaNs
      self.Labels = [''] * len(self.Data)
      for i in range(len(self.Data)):
        self.Labels[i] = str(i)
      return self.Data
    # Otherwise it's a tab-delimited file
    _Data = readDTFile(pf)
    # Check if result file
    rsltfile = True
    lend = len(_Data)
    lend2 = lend/2
    if ipfe == ".tab" or pife == ".tdf": # tab-delimited file extenstion
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
    # Read result file
    if rsltfile: # a known formatted file type is straightforward
      if type(spec) is bool: spec = 'MWD'
      if spec is None: spec = 'MWD'
      self.Data = [[]] * lend2
      self.Labels = [''] * lend2
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
          self.Labels[i] = str(i)
        else:
          if k > l:
            _label = _log[k]
            _label = _label.replace("setLabel('", "")
            _label = _label.replace("')", "")
          else:
            _label = _log[l]
            iq = strfind(_label, "'")
            _label = _label[iq[-2]+1:iq[-1]]
          self.Labels[i] = str(i) + " (" + _label + ")"
        _data = _Data[i*2+1]
        k = listfind(_data[0], spec)
        if k is None:
          raise ValueError("File " + ipfn + " missing headed data specification: " + spec)
        J = len(_data)
        self.Data[i] = np.tile(np.NaN, J-1)
        for j in range(1, J):
          _datum = _data[j][k]
          if isnumeric(_datum):
            self.Data[i][j-1] = float(_datum)
        self.Data[i] = nanravel(self.Data[i]) # remove NaNs
      for i in range(lend2-1, -1, -1): # remove empty data vectors
        if not(len(self.Data[i])):
          del self.Data[i]
          del self.Labels[i]
      return self.Data
    # Otherwise we have to assume a single-table formatted tab-delimited file
    # But we do not know the orientation
    if spec is None: spec = False
    if nDim(_Data) == 3:
      if len(_Data) == 1:
        _Data = _Data[0]
      elif isint(spec):
        _Data = _Data[spec]
      else: # check for singleton middle dimension
        maxn = 0
        for i in range(len(_Data)):
          maxn = max(maxn, len(_Data[i]))
        if maxn > 1:
          raise ValueError("Unrecognised data structure.")
        else:
          Data_ = [[]] * len(_Data)
          for i in range(len(_Data)):
            if len(_Data[i]):
              Data_[i] = _Data[i][0]
          for i in range(len(Data_)-1, -1, -1):
            if not(len(Data_[i])):
              del Data_[i]
          _Data = Data_
    elif nDim(_Data) == 1:
      _Data = [_Data]
    elif nDim(_Data) != 2:
      raise ValueError("Unrecognised data structure")
    if spec == True:
      try :
        _Data = listtranspose(_Data)
      except IndexError: # doesn't want to tranpose
        pass
    self.Data = readDTData(_Data)
    if self.Labels is None:
      nData = len(self.Data)
      self.Labels = [''] * nData
      for i in range(nData):
        self.Labels[i] = '#' + str(i)
    return self.Data
  def mpfa(self, fi = None, e = None, callback = None):
    if callback is None: callback = self.runbqa
    self.setData(self.Data, e, None)
    if fi is None: 
      self.fi = mp.figure()
      mp.subplot(1, 1, 1)
    else:
      self.fi = fi
    self.plotMoments()    
    mp.title(r"".join(("$\hat{Q} = ", str(self.parq), ", \hat{N} = ", str(self.parn), ", \epsilon = ", str(self.hate), "$")))
    self.axs = [[]] * 2
    self.btn = [[]] * 2
    self.axs[0] = exes('b', 1, 3)
    self.btn[0] = mpl.widgets.Button(self.axs[0], 'Help')
    self.btn[0].on_clicked(self.mpfahl)
    self.axs[1] = exes('b', 2, 3)
    self.btn[1] = mpl.widgets.Button(self.axs[1], 'Run Quantal Analysis')
    self.btn[1].on_clicked(callback)
    self.fi.canvas.draw()
    try:
      self.fi.show()
    except AttributeError:
      pass
  def mpfahl(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[1]+MANSUFF)
  def runbqa(self, spec = None, nmax = None, rescoef = 1, resmin = 128, resmax = 128):
    if not(isint(spec)): spec = 1
    modelbeta = spec == 2
    self.setLimits(None, None, None, None, nmax)
    self.setPriors(rescoef, resmin, resmax, modelbeta)
    self.calcPosts()
    self.showbqa()
  def showbqa(self, _fI = None, callback = None):
    if callback is None:
      callback = self.plotqlf
    if _fI is None:
      _fI = mp.figure()
    self.fI = _fI
    self.plotSummary(self.fI)
    btnlb = [r'Export $f(q)$', r'Export $f(\gamma)$', r'Export $f(r)$', r'Export $f(\alpha)$', r'Export $f(n)$',
        r'Export $f(v)$']
    btncb = [self.f_q, self.f_g, self.f_r, self.f_a, self.f_n, self.f_v]
    if not(self.modelBeta):
      del btnlb[3]
      del btncb[3]
    del self.btn
    nb = len(btnlb)
    self.axes = [[]] * (nb + 3)
    self.btns = [[]] * (nb + 3)
    for i in range(nb):
      self.axes[i] = exes('t', i, nb)
      self.btns[i] = mpl.widgets.Button(self.axes[i], btnlb[i])
      self.btns[i].on_clicked(btncb[i])
    btnLb = ['Help', 'Joint distributions', 'Amplitude distributions']
    btnCb = [self.bqahl, self.plotjmd, callback]
    i = nb
    for j in range(3):
      self.axes[i+j] = exes('b', j, 3)
      self.btns[i+j] = mpl.widgets.Button(self.axes[i+j], btnLb[j])
      self.btns[i+j].on_clicked(btnCb[j])    
    self.fI.canvas.draw()
    try:
      self.fI.show()
    except AttributeError:
      pass
  def bqahl(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[3]+MANSUFF)
  def plotjmd(self, ev = None, _fi_ = None):
    if _fi_ is None:
      _fi_ = mp.figure()
    self.fi_ = _fi_
    self.plotMarg2(self.fi_)
    self.fi_.canvas.draw()
    try:
      self.fi_.show()
    except AttributeError:
      pass
  def plotqlf(self, ev = None, _Fi = None, bw = None):
    if _Fi is None:
      _Fi = mp.figure()
    self.Fi = _Fi
    self.overlayHist(bw, self.Fi)
    self.qlfax = [[]] * 2
    self.qlfbt = [[]] * 2
    self.qlfax[0] = exes('b', 1, 3)
    self.qlfbt[0] = mpl.widgets.Button(self.qlfax[0], 'Help')
    self.qlfbt[0].on_clicked(self.plotqlfhl)
    self.qlfax[1] = exes('b', 2, 3)
    self.qlfbt[1] = mpl.widgets.Button(self.qlfax[1], 'Export model profiles')
    self.qlfbt[1].on_clicked(self.f_x)
    self.Fi.canvas.draw()
    try:
      self.Fi.show()
    except AttributeError:
      pass
  def plotqlfhl(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[5]+MANSUFF)
  def f_q(self, ev = None):
    self.export(self.q, self.PQ.P, '_q')
  def f_g(self, ev = None):
    self.export(self.g, self.PG.P, '_g')
  def f_r(self, ev = None):
    self.export(self.r, self.PR.P, '_r')
  def f_a(self, ev = None):
    self.export(self.a, self.PA.P, '_a')
  def f_n(self, ev = None):
    self.export(self.n, self.PQ.P, '_n')
  def f_v(self, ev = None):
    self.export(self.v, self.PG.P, '_v')
  def f_x(self, ev = None):
    self.export(self.HXY, None, '_x')
  def export(self, x, y = None, s = '_', transp = True): 
    xy = x if y is None else np.vstack((x,y))
    if transp:
        xy = xy.T
    _opfn = self.path + '/' + self.stem + s + ".tab"
    opfn_ = raw_input('Export [%s]: ' % _opfn)
    opfn = _opfn or opfn_
    writeDTFile(opfn, [xy])
    print("Exported to: " + opfn)

class Window(QtGui.QDialog): # A convenience class to embed matplotlib in Qt
  def __init__(self, parent=None, btnName = None, btnCall = None):
    if btnName == "": btnName = None
    super(Window, self).__init__(parent)
    self.figure = mp.figure()
    self.canvas = FigureCanvas(self.figure)
    self.toolbar = NavigationToolbar(self.canvas, self)
    if btnName is not None:
      self.button = QtGui.QPushButton(btnName)
      self.button.clicked.connect(btnCall)
    layout = QtGui.QVBoxLayout()
    layout.addWidget(self.toolbar)
    layout.addWidget(self.canvas)
    if btnName is not None:
      layout.addWidget(self.button)
    self.setLayout(layout)

class bqagui (bqa): # Convenience Qt-based GUI wrapper class for bqa
  def __init__(self):
    bqa.__init__(self)
  def openFile(self, spec = True):
    self.Base = lbw.LBWidget(None, None, None, 'base')
    self.pf = self.Base.dlgFileOpen("Open File", "", "Tab-delimited (*.tab);;XLS (*.xls);;XLSX (*.xlsx);;All (*.*)")
    rf =  self.readFile(self.pf, spec) 
    if rf is not None:
      self.initGUI()
  def readFile(self, pf = "", spec = True): #spec=True specifies col-arranged TAB data
    return bqa.readFile(self, pf, spec)
  def initGUI(self):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.EDx = lbw.LBWidget(self.Dlg, "Enter baseline standard deviation (leave as `e' for automated estimation): ", 1, 'edit', 'e')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"]) 
    self.BBx.Widgets[0].connect("btndown", self.blhelp)
    self.BBx.Widgets[1].connect("btndown", self.MPFA)
    self.Box.add(self.EDx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def blhelp(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[0]+MANSUFF)
  def MPFA(self):
    self.uie = self.EDx.retData()
    self.Dlg.close()
    self.uie = None if not(isnumeric(self.uie)) else float(self.uie)
    del self.Dlg
    self.win = Window(None, None, self.plotMPFA)
    self.win.show()
    self.plotMPFA()
  def plotMPFA(self, event = None):
    self.mpfa(self.win.figure, self.uie, self.RunBQA)
  def RunBQA(self, ev = None):
    self.imv = [[]] * self.NX
    for i in range(self.NX):
      self.imv[i] = "".join( (self.Labels[i], ": Mean=", str(self.mn[i]), "; Var.=", str(self.vr[i])) )
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBi = lbw.LBWidget(self.Dlg, "Data selection", 1,"listbox", None, self.imv)
    self.LBi.setMode(3, range(self.NX))
    self.EDn = lbw.LBWidget(self.Dlg, "Max. no. release sites (leave as `e' for automated estimation) ", 1, 'edit', 'e')
    self.EDs = lbw.LBWidget(self.Dlg, 'Probability of release sampling resolution: ', 1, 'edit', '128')
    self.EDv = lbw.LBWidget(self.Dlg, 'Coefficient of varation sampling resolution: ', 1, 'edit', '128')
    self.EDa = lbw.LBWidget(self.Dlg, "Heterogeneity sampling resolution (leave as `1' for homogeneous model): ", 1, 'edit', '1')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button", "Button"], ["Help", "Cancel", "OK"]) 
    self.BBx.Widgets[0].connect("btndown", self.BQAHL)
    self.BBx.Widgets[1].connect("btndown", self.BQACC)
    self.BBx.Widgets[2].connect("btndown", self.BQAOK)
    self.Box.add(self.LBi)
    self.Box.add(self.EDn)
    self.Box.add(self.EDs)
    self.Box.add(self.EDv)
    self.Box.add(self.EDa)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def BQAHL(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[2]+MANSUFF)
  def BQACC(self, ev = None):
    self.Dlg.close()
    del self.Dlg
  def BQAOK(self, ev = None):
    uimv = self.LBi.retData()
    OK = True
    OK = OK and len(uimv) > 1
    OK = OK and self.EDn.valiData(1, 0)
    OK = OK and self.EDs.valiData(1, 1)
    OK = OK and self.EDv.valiData(1, 1)
    OK = OK and self.EDa.valiData(1, 1)
    if not(OK): return
    maxn = self.EDn.retData()
    maxn = None if not(isnumeric(maxn)) else float(maxn)
    limn = maxn if maxn is None else [1, maxn]
    sres = abs(int(self.EDs.retData()))
    vres = abs(int(self.EDv.retData()))
    ares = abs(int(self.EDa.retData()))
    self.Dlg.close()
    del self.Dlg
    _X = [[]] * len(uimv)
    for i in range(len(uimv)):
      _X[i] = nanravel(self.X[uimv[i]])
    modelbeta = ares > 1
    res = [sres, vres, ares] if modelbeta else [sres, vres]
    self.setData(_X, self.e, maxn)
    self.setLimits(None, None, None, limn, maxn)
    self.setPriors(res, None, None, modelbeta)
    self.calcPosts()
    self.Win = Window(None, None, self.plotBQA)
    self.Win.show()
    self.plotBQA()
  def plotBQA(self, ev = None):
    self.showbqa(self.Win.figure, self.plotQLF)
  def plotQLF(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.EDx = lbw.LBWidget(self.Dlg, "Enter bin width (leave as `o' for optimal size): ", 1, 'edit', 'o')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Help", "OK"]) 
    self.BBx.Widgets[0].connect("btndown", self.PlotQLFHL)
    self.BBx.Widgets[1].connect("btndown", self.PlotQLF)
    self.Box.add(self.EDx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def PlotQLFHL(self, ev = None): webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[4]+MANSUFF)
  def PlotQLF(self, ev = None):
    self.uibw = self.EDx.retData()
    self.Dlg.close()
    self.uibw = None if not(isnumeric(self.uibw)) else float(self.uibw)
    del self.Dlg
    self.Wind = Window(None, None, self.plotqlf)
    self.Wind.show()
    self.plotqlf(ev, self.Wind.figure, self.uibw)
  def plotjmd(self, ev = None):
    self.Wind_ = Window(None, None, None)
    self.Wind_.show()
    bqa.plotjmd(self, ev, self.Wind_.figure)
  def export(self, x, y = None, s = '_', transp = True): 
    xy = x if y is None else np.vstack((x,y))
    if transp:
        xy = xy.T
    _opfn = self.path + '/' + self.stem + s + ".tab"
    senderwidget = lbw.LBWidget(None, None, None, 'base')                        
    pf = senderwidget.dlgFileSave("Save File", _opfn, '*.tab')
    if len(pf):
      writeDTFile(pf, [xy])

