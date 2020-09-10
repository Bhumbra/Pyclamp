# A nodal network module

import numpy as np 
import scipy as sp
import scipy.special as special

DEFAULT_LD = 0.05
EPSILON_FP = 1e-300

#-----------------------------------------------------------------------------------------------------------------------
def ksd(X, *args, **kwds): # Kolmogorov-Smirnov distances (sorted, not max)
  Y = np.sort(X, *args, **kwds)
  sY = np.array(Y.shape, dtype = int)
  N = len(sY)
  means = np.mean(Y, *args, **kwds)
  stdvs = np.std(Y, *args, **kwds)
  if type(means) is float: means = np.array([means])
  if type(stdvs) is float: stdvs = np.array([stdvs])
  sm = np.array (means.shape, dtype = int)
  if len(sm) != N:
    while len(sm) != N: sm = np.hstack((1, sm))
    means = means.reshape(sm)
    stdvs = stdvs.reshape(sm)
  res = sm
  rep = sY / np.array(means.shape, dtype = int)
  i = np.nonzero(rep > 1)[0]
  n = np.prod(rep[i])
  count = np.arange(n, dtype = float) + 0.5
  Means = np.tile(means.reshape(sm), rep)
  Stdvs = np.tile(stdvs.reshape(sm), rep)
  o = np.ones(N, dtype = int)
  o[i] = rep[i]
  r = sY / o
  obcum = np.tile(count.reshape(o), r) / float(n)
  excum = 0.5 * (1. + special.erf( (Y - Means) / ((Stdvs + EPSILON_FP)*np.sqrt(2.))))
  return np.fabs(obcum - excum)

#-----------------------------------------------------------------------------------------------------------------------

def sech(x): #bizarre but true: there's no np.sech
  return 1./(np.cosh(x)+EPSILON_FP)

def scaleoffset(x, coefficient = 1., offset = None, opts = None): # opts = None, 0, or False: (mx+c),  otherwise (mx+c, mx)
  if opts is None: opts = False
  mx = coefficient * x
  mxc = mx if offset is None else mx + offset
  if not(opts): return mxc
  return mxc, mx

def altperm(_n):
  odd = _n % 2
  n = _n - 1 if odd else _n
  i = np.array([], dtype = int)
  if _n == 1:
    i = np.array([0], dtype = int)
  elif n > 1:
    n2 = n/2
    i = np.arange(0, n2, dtype=int)
    i = np.ravel(np.vstack((i,i+n2)).T)
    if odd: i = np.hstack((i,n))
  return i
  
def GHA(W, x, y, *args):
  return y*x.T - np.tril(y*y.T)*W 
  '''
  r, c = W.shape()
  D = np.empty( (r, c), dtype = float)
  for i in range(r):
    D[i] = y[i] * x.T - np.dot(np.tile(y[i]*y[i]', (1, c)), W[i])
  return D
  '''

def anneal(_i = 0, _spec = DEFAULT_LD):
  if type(_spec) is float:
    return float(_spec)
  else:
    spec = np.array(_spec, dtype = float)
    if len(spec) == 1: 
      return float(spec[0])
    i = float(_i)
    if len(spec) == 2:
      I = spec[0]/(spec[1]+EPSILON_FP)
      spec = np.array([spec[0], spec[1], spec[0]-spec[1]*(I-1.)])
    S = spec[0] - spec[1]*i
    if S >= spec[2]: return S
    return spec[2]

#-----------------------------------------------------------------------------------------------------------------------
class layer: # class for single nodal layer
  ld_func = None # learning rate changes (lambda)
  ip_func = None # input (None denotes a = W*x)
  at_func = None # activation transfer (None denotes equality and changes input & output args of learning rule)
  lr_func = None # learning rule: D = rule(W, x, y) if at_func is None else D, d = rule(W, w, x, a, y, Wx)
  pl_func = None # post-learning weight changing function
  Data = None
  X = None
  def __init__(self, _centrescale = 1): # -1 is off; otherwise axis
    self.setSC(_centrescale)
    self.setDF()
    self.setLD()
    self.setIP()
    self.setAT()
    self.setLR()
    self.setPL()
    self.setData()
  def setSC(self, _centrescale = -1):
    if type(_centrescale) is int: _centrescale = [_centrescale, _centrescale]
    self.cendim, self.scadim = _centrescale
  def setDF(self, _ld_dfun = anneal, _ip_dfun = None, _at_dfun = None, _lr_dfun = GHA, _pl_dfun = None):
    self.ld_dfun = _ld_dfun
    self.ip_dfun = _ip_dfun
    self.at_dfun = _at_dfun
    self.lr_dfun = _lr_dfun
    self.pl_dfun = _pl_dfun
  def setLD(self, _ld_func = None, *_ld_args, **_ld_kwds):
    if _ld_func is None: _ld_func = self.ld_dfun
    self.ld_func = _ld_func
    self.ld_args = _ld_args
    self.ld_kwds = _ld_kwds
    if type(self.ld_func) is tuple: self.ld_func = list(self.ld_func)
    if self.ld_func is None: return
    if hasattr(self.ld_func, '__call__'): return
    if type(self.ld_func) is int:  self.ld_func = float(self.ld_func)
    self.ld_args = tuple([self.ld_func]) if type(self.ld_func) is float else list(self.ld_func),
    self.ld_func = self.ld_dfun
  def setIP(self, _ip_func = None, *_ip_args, **_ip_kwds):
    if _ip_func is None: _ip_func = self.ip_dfun
    self.ip_func = _ip_func
    self.ip_args = _ip_args
    self.ip_kwds = _ip_kwds 
  def setAT(self, _at_func = None, *_at_args, **_at_kwds):
    if _at_func is None: _at_func = self.at_dfun
    self.at_func = _at_func
    self.at_args = _at_args
    self.at_kwds = _at_kwds 
  def setLR(self, _lr_func = None, *_lr_args, **_lr_kwds):
    if _lr_func is None: _lr_func = self.lr_dfun
    self.lr_func = _lr_func
    self.lr_args = _lr_args
    self.lr_kwds = _lr_kwds 
  def setPL(self, _pl_func = None, *_pl_args, **_pl_kwds):
    if _pl_func is None: _pl_func = self.pl_dfun
    self.pl_func = _pl_func
    self.pl_args = _pl_args
    self.pl_kwds = _pl_kwds 
  def process(self, X, **kwargs): # plagiarised from my own fpfunc.censca
    Y = np.array(X, dtype = float)
    sY = list(Y.shape)
    if len(sY) == 1: sY.append(1)
    oY = [1] * len(sY)
    i, j = self.cendim, self.scadim
    if i >= 0: 
      res = sY[:]; res[i] = 1
      rep = oY[:]; rep[i] = sY[i]
      self.means = np.mean(Y, axis = i, **kwargs)
      Y -= np.tile(self.means.reshape(res), rep)
    if j >= 0:
      res = sY[:]; res[j] = 1
      rep = oY[:]; rep[j] = sY[j]
      self.stdvs = (np.std(Y, axis = j, **kwargs))
      Y /= np.tile(self.stdvs.reshape(res)+EPSILON_FP, rep)
    return Y           
  def setData(self, _Data = None, transpose = False):
    if _Data is None: return
    self.Data = self.process(_Data)
    self.X = np.matrix(self.Data)
    if transpose: self.X = self.X.T
    self.mn = np.mean(self.X, axis = 1)
    self.sd = np.std(self.X, axis = 1)
    self.s0 = -0.5+np.matrix(np.eye(len(self.sd)))*np.mean(self.sd)
  def learn(self, N = 1, W0 = None, w0 = None, pgb = None, pgbtit = "learning..."): # polymorphism-friendly
    if self.X is None: raise ValueError("Must invoke self.setData(X) before self.learn")
    if W0 is None: W0 = self.s0
    if w0 is None: w0 = -self.mn
    m, n = self.X.shape
    self.n = n
    W, w = np.matrix(W0, dtype = float), np.matrix(w0, dtype = float)
    self.D, self.d = np.zeros(N, dtype = float), np.zeros(N, dtype = float)
    Wx = None
    k = -1
    if pgb is not None: pgb.init(pgbtit, N*n)
    done = N == 0
    h = -1
    Xh = np.matrix(self.X, copy=True)
    while not(done):
      h += 1
      Wh = np.copy(W)
      wh = np.copy(w)
      Xh = np.matrix(Xh[:, altperm(n)])
      self.ld = self.ld_func if type(self.ld_func) is float else self.ld_func(h, *self.ld_args, **self.ld_kwds)
      for i in range(n):
        k += 1
        if pgb is not None: pgb.set(k)
        x = Xh[:, i]
        if self.ip_func is None:
          a = W * x
        else:
          a, Wx = self.ip_func(x, W, w, True, *self.ip_args, **self.ip_kwds)
        if self.at_func is None:
          y = a
          dW = self.lr_func(W, x, y)
        else:
          y = self.at_func(a, *self.at_args, **self.at_kwds)
          dW, dw = self.lr_func(W, w, x, a, y, Wx, *self.lr_args, **self.lr_kwds)
          w += self.ld*dw
        W += self.ld*dW
      self.D[h] = np.sum(np.fabs(W-Wh))
      self.d[h] = np.sum(np.fabs(w-wh))
      done = h == N - 1 # or (self.D[h] <= EPSILON_FP and self.d[h] <= EPSILON_FP)
    if pgb is not None: pgb.set(N*n-1, True)
    if pgb is not None: pgb.close()
    H = 1 if self.pl_func is None else 2
    for h in range(H):
      if not(h):
        self.W = np.matrix(W)
        self.w = np.matrix(w)
      if self.ip_func is None:
        self.A = self.W * self.X
      else:
        self.A, WX = self.ip_func(self.X, self.W, self.w, True, *self.ip_args, **self.ip_kwds)
      if self.at_func is None:
        self.Y = self.A
      else:
        self.Y = self.at_func(self.A, *self.at_args, **self.at_kwds)
      if not(h) and self.pl_func is not None:
        self.pl_func(self, *self.pl_args, **self.pl_kwds)
    if self.at_func is None: return W
    return W, w

#-----------------------------------------------------------------------------------------------------------------------
# ACTIVATION TRANSFER FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------

def I1AT(a): # logistic for supergaussian data
  return 1./(1.+np.exp(-a))

def I2AT(a): # tanh for supergaussian data
  return np.tanh(a)


IATF = [np.negative, I1AT, I2AT] #  activation transfer functions

#-----------------------------------------------------------------------------------------------------------------------
# BASIC INFOMAX LEARNING RULE FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------
def I1LR(W, w, x, a, y, *args): # basic infomax learning rule using logistic activation function
  d = 1. - 2.*y
  D = np.linalg.inv(W.T) + d * x.T
  return D, d

def I2LR(W, w, x, a, y, *args): # basic infomax learning rule using tanh activation function
  d = -2.*y
  D = np.linalg.inv(W.T) + d * x.T
  return D, d

IMLR = [None, I1LR, I2LR, None] # Basic infomax learning rules

#-----------------------------------------------------------------------------------------------------------------------

class imlayer (layer): # basic infomax nodal network layer
  ATF = None # Activation transfer function: 1 = logistic, 2 = tanh
  def __init__(self, _centrescale = 1, _ATF = 1):
    layer.__init__(self, _centrescale)
    self.setATF(_ATF)
  def setATF(self, _ATF = 2):
    self.ATF = _ATF
    self.initIM()
  def initIM(self): # initialise infomax
    self.setIP(scaleoffset)
    self.setAT(IATF[self.ATF])
    self.setLR(IMLR[self.ATF])
    if self.lr_func is None:
      print("Warning: invalid learning rule for infomax specification")
  def learn(self, N = 1, W0 = None, w0 = None, pgb = None, pgbtit = "Learning..."): # polymorphism-friendly
    if self.X is None: raise ValueError("Must invoke self.setData(X) before self.learn")
    if W0 is None: [_0, _1, W0] = np.linalg.svd(self.X.T)    
    if w0 is None: w0 = -self.mn
    return layer.learn(self, N, W0, w0, pgb, pgbtit)

#-----------------------------------------------------------------------------------------------------------------------
# EXTENDED INFOMAX ACTIVATION TRANSFER FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------

def N1AT(a): # logistic for subgaussian data (same as subgaussian)
  return 1./(1.+np.exp(-a))

def N2AT(a): # tanh for subgaussian data
  return -2.*(np.tanh(a) - np.tanh(a+2.) - np.tanh(a-2.))

def N3AT(a):  # sloped logistic for subgaussian data
  return a - np.tanh(a)

def P1AT(a): # logistic for supergaussian data
  return 1./(1.+np.exp(-a))

def P2AT(a): # tanh for supergaussian data
  return np.tanh(a)

def P3AT(a):  # sloped logistic for supergaussian data
  return a + np.tanh(a)

NATF = [np.copy,     N1AT, N2AT, N3AT] # subgaussian activation transfer functions
PATF = [np.negative, P1AT, P2AT, P3AT] # supergaussian activation transfer functions

#-----------------------------------------------------------------------------------------------------------------------
# EXTENDED INFOMAX LEARNING RULE FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------

def N1LR(W, w, x, a, y, *args): # extended infomax subgaussian learning rule using logistic activation function
  I = np.matrix(np.eye(len(y)))
  d = 1. - 2.*y
  m2aT = -2.*a.T
  D = (I - np.tanh(a)*m2aT + np.tanh(a+2.)*m2aT + np.tanh(a-2.)*m2aT) * W
  return D, d

def N2LR(W, w, x, a, y, *args): # extended infomax subgaussian learning rule using tanh activation function
  I = np.matrix(np.eye(len(y)))
  d = -2.*y
  m2aT = -2.*a.T
  D = (I - np.tanh(a)*m2aT + np.tanh(a+2.)*m2aT + np.tanh(a-2.)*m2aT) * W
  return D, d

def N3LR(W, w, x, a, y, *args): # extended infomax subgaussian learning rule using sloped logistic activation function
  I = np.matrix(np.eye(len(y)))
  d = -2.*y
  D = (I + np.tanh(a)*a.T - a*a.T) * W
  return D, d

def P1LR(W, w, x, a, y, *args): # extended infomax supergaussian learning rule using logistic activation function
  I = np.matrix(np.eye(len(y)))
  d = 1. - 2.*y
  D = (I - 2.*y*a.T) * W
  return D, d

def P2LR(W, w, x, a, y, *args): # extended infomax supergaussian learning rule using tanh activation function
  I = np.matrix(np.eye(len(y)))
  d = -2.*y
  D = (I - 2.*y*a.T) * W
  return D, d

def P3LR(W, w, x, a, y, *args): # extended infomax supergaussian learning rule using sloped logistic activation function
  I = np.matrix(np.eye(len(y)))
  d = -2.*y
  D = (I - np.tanh(a)*a.T - a*a.T) * W
  return D, d

EILR = [[None, I1LR, I2LR, None], # Basic infomax
        [None, N1LR, N2LR, N3LR], # Extended infomax for subgaussian
        [None, P1LR, P2LR, P3LR]] # Extended infomax for supergaussian

#-----------------------------------------------------------------------------------------------------------------------
class eimlayer (imlayer): # extended infomax nodal network layer
  b = None
  c = None
  _c = None
  Esqrda = None
  ATF = None # Activation transfer function: 1 = logistic, 2 = tanh, 3 = slogistic
  EIK = None # Extended infomax kurtosis: 0 = basic informax otherwise extended informax:
             # -1 = subgaussian, 1 = supergaussian, 2 = asymptotic stability nonlinearity switching criterion
  def __init__(self, _centrescale = 1, _ATF = 1, _EIK = 0):
    layer.__init__(self, _centrescale)
    self.setATF(_ATF)
    self.setEIK(_EIK)
  def setATF(self, _ATF = 2):
    self.ATF = _ATF
    if self.EIK is not None: self.initIM()
  def setEIK(self, _EIK = 0):
    self.EIK = _EIK
    if self.ATF is not None: self.initIM()
  def initIM(self): # initialise infomax
    self.setIP(scaleoffset)
    self.setPL(self.postLearn)
    if not(self.EIK): # basic infomax
      self.setAT(PATF[self.ATF])
      self.setLR(EILR[0][self.ATF])
    elif self.EIK == -1: # extended infomax assuming subgaussian
      self.setAT(NATF[self.ATF])
      self.setLR(EILR[1][self.ATF])
    elif self.EIK == 1: # extended infomax assuming supergaussian
      self.setAT(PATF[self.ATF])
      self.setLR(EILR[2][self.ATF])
    else: # extended infomax with stability switching criterion
      AT_ = [None, self.AT1, self.AT2, self.AT3]
      LR_ = [None, self.LR1, self.LR2, self.LR3]
      self.setAT(AT_[self.ATF])
      self.setLR(LR_[self.ATF])
    if self.lr_func is None:
      print("Warning: invalid learning rule for infomax specification")
  def postLearn(self, _self, *args, **kwds):
    # Restore subgaussian/supergaussian flags
    if self.EIK == -1:
      self.c = np.matrix(-np.ones(self.n, dtype = float))
    elif self.EIK == 1:
      self.c = np.matrix(np.ones(self.n, dtype = float))
    else:
      self.c = self._c
    self.b = 1.-self.c
    # Order row weights by mean KS-distance
    md = np.ravel(np.mean(ksd(self.A, axis = 1), axis = 1))
    i = np.argsort(md)[::-1]
    self.md = md[i]
    self.W = self.W[i]
    if self.at_func is not None:
      self.w = np.matrix(self.w[i])
  def setAE(self, a = None):
    if a is None:
      self.Counter = 0
      self.Esqrda = None
      self.Etanha = None
      self.Esech2 = None
      self.Ssqrda = None
      self.Stanha = None
      self.Ssech2 = None
      self.I = None
      self.c = None
      return
    self.Esqrda = np.multiply(a, a)
    self.Etanha = np.multiply(np.tanh(a), a)
    secha = sech(a)
    self.Esech2 = np.multiply(secha, secha)
    if not(self.Counter):
      self.I = np.matrix(np.eye(len(a)))
      self.Ssqrda = self.Esqrda
      self.Stanha = self.Etanha
      self.Ssech2 = self.Esech2
      self.Counter = 1
      return
    self.Ssqrda += self.Esqrda
    self.Stanha += self.Etanha
    self.Ssech2 += self.Esech2
    self.Counter += 1
    rC = 1./ float(self.Counter)
    self.Esqrda = self.Ssqrda * rC
    self.Etanha = self.Stanha * rC
    self.Esech2 = self.Ssech2 * rC
  def setCB(self, a):
    if self.c is None: 
      self.setAE() # initialise expectation terms
      self.c = np.matrix(np.ones(len(a), dtype = float)).T
      self.b = 1.-self.c
      return
    if self.Esqrda is not None:
      self.c = np.sign(np.multiply(self.Esech2, self.Esqrda) - self.Etanha)
      self.b = 1.-self.c
      self._c = self.c
  def AT1(self, a): # logistic
    self.setCB(a)
    return 1./(1.+np.exp(-a))
  def AT2(self, a): # tanh
    self.setCB(a)
    return -2.*(np.tanh(a) - np.tanh(a+self.b) - np.tanh(a-self.b))
  def AT3(self, a): # slope logistic
    self.setCB(a)
    return a + np.multiply(self.c, np.tanh(a))
  def LR1(self, W, w, x, a, y, *args):
    self.setAE(a)
    d = 1. - 2.*y
    m2aT = -2.*a.T
    D = (self.I - np.tanh(a)*m2aT + np.tanh(a+self.b)*m2aT + np.tanh(a-self.b)*m2aT) * W
    if self.Counter == self.X.shape[1]: self.setAE()
    return D, d
  def LR2(self, W, w, x, a, y, *args):
    self.setAE(a)
    d = - 2.*y
    m2aT = -2.*a.T
    D = (self.I - np.tanh(a)*m2aT + np.tanh(a+self.b)*m2aT + np.tanh(a-self.b)*m2aT) * W
    if self.Counter == self.X.shape[1]: self.setAE()
    return D, d
  def LR3(self, W, w, x, a, y, *args):
    self.setAE(a)
    d = - 2.*y
    D = (self.I - np.multiply(self.c, np.tanh(a)*a.T) - a*a.T) * W
    if self.Counter == self.X.shape[1]: self.setAE()
    return D, d

