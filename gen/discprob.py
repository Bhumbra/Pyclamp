#pgb A probability module

import numpy as np
import scipy as sp
import scipy.stats as stats
from fpfunc import *
from sifunc import *
from lsfunc import *
import warnings

DEFPMAT = False
npmatrix=np.matrixlib.defmatrix.matrix

#-----------------------------------------------------------------------------------------------------------------------
def zpdf(x, m = 0., v = 1., c = None):
  if c is None:
    return np.exp( - (x-m)**2./(2.*v))/np.sqrt(2.*np.pi*v)
  return np.exp( - (x-m)**2./(2.*v)) * c

#-----------------------------------------------------------------------------------------------------------------------
def glnf(g):
  return sp.special.gammaln(g)

#-----------------------------------------------------------------------------------------------------------------------
def gpdf(_x, _g = 1., _l = 1.):
  mino = 1e-300
  maxo = 1e300
  x = maxo
  if type(_x) is np.ndarray:
    x = _x.copy() + mino
    x[x < 0] = maxo
  elif _x >= 0:
    x = np.array(_x+mino, dtype = float).reshape(1)
  g = _g + mino
  l = _l + mino
  y = np.exp(- x/l - g * np.log(l) - sp.special.gammaln(g) + (g-1.0)*np.log(x) )
  y[np.isnan(y)] = 0.0
  return y

#-----------------------------------------------------------------------------------------------------------------------
def gampdf(_x, _g, _l):
  mino = 1e-300
  maxo = 1e300
  x = maxo
  if type(_x) is np.ndarray:
    x = _x.copy() + mino
    x[x < 0] = maxo
  elif _x >= 0:
    x = np.array(_x+mino, dtype = float).reshape(1)
  g = _g + mino
  l = _l + mino
  y = np.exp(- x/l - g * np.log(l) - sp.special.gammaln(g) + (g-1.0)*np.log(x) )
  y[np.isnan(y)] = 0.0
  return y

#-----------------------------------------------------------------------------------------------------------------------
def mnpdf(_x, _p) :
  mino = 1e-300
  x, p = np.array(_x, dtype = float), np.array(_p, dtype = float)
  y = np.exp(sp.special.gammaln(x.sum()+1.0) - sum(sp.special.gammaln(x+1.0)) + sum(x * np.log(p+mino)))
  return y

#-----------------------------------------------------------------------------------------------------------------------
def betaentropy(a, b):
  return sp.special.gammaln(a)+sp.special.gammaln(b)-sp.special.gammaln(a+b) - (a - 1.) * (sp.special.digamma(a) - sp.special.digamma(a+b)) - (b - 1.)*(sp.special.digamma(b) - sp.special.digamma(a+b))  

#-----------------------------------------------------------------------------------------------------------------------
def betanomial(_n = 1.0, _a = 1.0, _b = 1.0, _pgb = False):
    onem = 1e-11
    n, a, b = int(_n), np.array(_a, dtype = float), np.array(_b, dtype = float)
    m = n + 1
    s = a.shape
    if not(np.all(s == b.shape)):
      raise ValueError('alpha and beta dimensions incommensurate,')
    #c = (np.arange(n, dtype = float)+0.5)/(float(n))
    c = np.linspace(onem, 1.-onem, n)
    S = np.hstack( (s, m) )
    P = np.zeros(S, dtype = float)
    a, b, P = a.ravel(), b.ravel(), P.ravel()
    N = len(a)
    p2 = 2**n
    POI = np.empty( (p2, n), dtype = bool )
    for i in range(p2):
      POI[i] = int2bool(i, n)    
    POI = POI.T
    QOI = np.logical_not(POI)
    PQ = np.empty( (n, p2), dtype = float )
    C = POI.sum(axis = 0)
    I = [[]] * m    
    for i in range(m):
      I[i] = np.nonzero(C == i)  
    if _pgb is not None: _pgb.init("Sampling beta distributions", N)
    k = 0 
    for i in range(N):
      if _pgb is not None: _pgb.set(i)
      #p = stats.beta.ppf(c, a[i], b[i])
      p = betasample(n, a[i], b[i], True)
      q = 1. - p
      #'''
      for j in range(n):
        PQ[j, POI[j, :]] = p[j]
        PQ[j, QOI[j, :]] = q[j]
      #'''  
      '''  
      p = np.tile(p.reshape(n, 1), (1, p2))
      q = np.tile(q.reshape(n, 1), (1, p2))     
      
      PQ[POI] = p[POI]
      PQ[QOI] = q[QOI]
      '''
      PQP = PQ.prod(axis = 0)
      pk = np.zeros(m, dtype = float)
      for j in range(m):
        pk[j] = PQP[I[j]].sum()
      P[k:k+m] = pk.copy()
      k += m
    if _pgb is not None: _pgb.reset()
    return P.reshape(S)  

#-----------------------------------------------------------------------------------------------------------------------
def betabinopmf(_i = 0.0, _n = 1.0, _a = 1.0, _b = 1.0):
  mino = 1e-300
  i, n = np.array(_i, dtype = float), np.array(_n, dtype = float)
  a, b = np.array(_a, dtype = float), np.array(_b, dtype = float)
  #y = np.exp(sp.special.gammaln(n + 1.)-sp.special.gammaln(i + 1.)-sp.special.gammaln(n - i + 1.))*sp.special.beta((a + i),(b + n - i))/sp.special.beta(a,b);
  #return y
  lognum = sp.special.gammaln(n+1.0) + sp.special.gammaln(a+i) + sp.special.gammaln(n+b-i) + sp.special.gammaln(a+b)
  logden = sp.special.gammaln(i+1.0) + sp.special.gammaln(n-i+1.0) + sp.special.gammaln(a+b+n) + sp.special.gammaln(a) + sp.special.gammaln(b)
  return np.exp(lognum - logden)  

#-----------------------------------------------------------------------------------------------------------------------
def betasample(_c = 1, alpha = None, beta = None, fixmom1 = False):
  mino = 1e-300
  eps1 = 1e-11
  onem = 1. - eps1
  if alpha is None:
    raise ValueError("alpha must always be entered.")
  elif not(beta is None):
    mu = alpha / (alpha + beta + mino)
  elif beta is None:
    mu = fixmom1
    beta = alpha/(mu+mino) - alpha
    fixmom1 = True
  if type(_c) is float: _c = np.array(_c, dtype = float)
  c = (np.arange(_c, dtype = float)+0.5)/(float(_c)) if isint(_c) else nanravel(_c)
  #if type(_c) is int: # (np.arange(_c, dtype = float)+0.5)/(float(_c)) is not so good here
  #  c = np.linspace(eps1, 1.-eps1, _c)
  n = len(c)
  x = stats.beta.ppf(c, alpha, beta)
  done = not(fixmom1)
  counter = 0
  while not(done):
    x += mu - x.mean()
    if x.min() < mino or x.max() > onem: 
      x[x < mino] = mino
      x[x > onem] = onem
      counter += 1
      done = counter >= n 
    else:
      done = True
  return x      

#-----------------------------------------------------------------------------------------------------------------------
def betanompmf(_n = 1.0, _a = 1.0, _b = 1.0, _pgb = None):
  n, a, b = nparray(_n, dtype = int), nparray(_a, dtype = float), nparray(_b, dtype = float)
  S, _S = a.shape, b.shape
  if S != _S: return ValueError("Alpha and beta inputs incommensurate")
  ni, a, b = n+1, a.ravel(), b.ravel()
  N, M = len(n), len(a)
  B = [None] * N
  m = np.array(2**n, dtype = int)
  ni = n + 1
  cm = list(np.array(np.cumsum(m), dtype = int))
  if _pgb is not None: _pgb.init("Sampling beta distributions", cm[-1])
  for h in range(N):
    if _pgb is not None: _pgb.set(cm[h], True)
    P = np.empty((m[h], n[h]), dtype = bool)
    for i in range(m[h]):
      P[i] = int2bool(i, n[h])
    P = P.T
    Q = np.logical_not(P)
    C = P.sum(axis = 0)
    I = [None] * ni[h]
    for i in range(ni[h]):
      I[i] = np.nonzero(C == i)
    A = np.empty(M*ni[h], dtype = float)
    c = cumspace(n[h])
    pq = np.empty((n[h], m[h]), dtype = float)
    k = 0
    for j in range(M):
      p = stats.beta.ppf(c, a[j], b[j])
      q = 1. - p
      for i in range(n[h]):
        pq[i, P[i]] = p[i]
        pq[i, Q[i]] = q[i]
      PQ = pq.prod(axis = 0)
      s = np.zeros(ni[h], dtype = float)
      for i in range(ni[h]):
        s[i] = PQ[I[i]].sum()
      A[k:k+ni[h]] = s.copy()
      k += ni[h]
    B[h] = A.copy() if M == 1 else A.reshape(np.hstack((S, ni[h]))).copy()
  if _pgb is not None: _pgb.reset()
  if isint(_n): return B[0]
  return B

#-----------------------------------------------------------------------------------------------------------------------
class mass:
  P = None
  '''
  def __init__(self, _P = None, _X = None, _pmat = DEFPMAT):
    self.setPdim()
    self.setPmat(_pmat)
    self.setPX(_P, _X)
  '''
  def __init__(self, *args):
    _P, _X, _pmat = None, None, DEFPMAT
    boolarg = False
    if len(args) == 1:
      if type(args[0]) is bool:
        _pmat = args[0]
        boolarg = True
      if not(boolarg):
        _P = args[0]
    elif len(args) == 2:
      if type(args[1]) is bool:
        _P, _pmat = args[0], args[1]
        boolarg = True
      if not(boolarg):
        _P, _X = args[0], args[1]
    elif len(args) == 3:
        _P, _X, _pmat = args[0], args[1], args[2]
    elif len(args) > 3:
      raise ValueError("Too many input arguments: " + str(len(args)))
    self.setPdim()
    self.setPmat(_pmat)
    self.setPX(_P, _X)
  def setPmat(self, _pmat = False):
    self.pmat = _pmat
    if self.P is None: return
    if not(self.NP): return
    if self.pmat and type(self.P) is np.ndarray:
      self.P = np.matrix(self.P.reshape((self.Pr, self.Pc)))
      self.setPdim((self.Pr, self.Pc))
    if not(self.pmat) and type(self.P) is npmatrix:
      self.P = np.array(self.P).reshape(self.nP)
      self.setPdim(list(self.nP))
  def setPX(self, _P = None, _X = None):
    _x = self.setP(_P, _X)
    if _X is None: _X = _x
    self.setX(_X)
    self.chkPX()
  def setPdim(self, _nP = None):
    # _nP is None: initialise
    # _nP is a list: P is an array
    # _nP is a tuple: P is a matrix of shape (_nP[0], _nP[1])
    # _nP is a ndarray: P is a matrix with array-equivalent dimensions of _nP

    if _nP is None:
      self.nP = np.empty(0, dtype = int)
      self.cP = np.empty(0, dtype = int)
      self.nR = np.empty(0, dtype = int)
      self.cR = np.empty(0, dtype = int)
      self.rR = np.empty(0, dtype = int)
      self.iR = np.empty(0, dtype = int)
      self.IR = np.empty(0, dtype = int)
      self.NP = 0
      self.np = 0
      self.Pr = 0
      self.Pc = 0
      self.nr = 0
      return None

    tnP = type(_nP)

    if tnP is tuple:
      if len(_nP) != 2: raise ValueError("Matrix dimension specification requires two-length tuple.")
      self.Pr, self.Pc = _nP[0], _nP[1]
      if self.pmat and self.NP and len(self.cP) and len(self.nP):
        _Pr = 1 if self.NP < 2 else self.cP[-2]
        _Pc = self.nP[-1]
        if self.Pr != _Pr: raise ValueError("Row dimension of probability matrix incommensurate with input.")
        if self.Pc != _Pc: raise ValueError("Colum dimension of probability matrix incommensurate with input.")
      return

    if tnP is list or tnP is np.ndarray:
      self.nP = np.copy(_nP)
      self.cP = np.cumprod(self.nP)
      self.np = self.cP[-1]
      self.NP = len(self.nP)
      self.NR = max(1, self.NP - 1)
      self.nr = 1
      self.nR = np.ones(1, dtype = int)
      self.cR = np.ones(1, dtype = int)
      self.rR = np.ones(1, dtype = int)
      self.iR = np.ones(1, dtype = int)
      if self.NR > 0:
        self.nR = self.nP[:-1]
        self.cR = np.hstack((1, np.cumprod(self.nR)))
        self.rR = np.hstack((np.cumprod(self.nR[::-1])[::-1], 1))
        self.nr = self.cR[-1]
        self.iR = np.array(self.nr / self.nR, dtype = int)
      _Pr = 1 if self.NP < 2 else self.cP[-2]
      _Pc = self.nP[-1]
      if not(self.Pr) or not(self.Pc):
        self.Pr = _Pr
        self.Pc = _Pc
      elif self.pmat and self.P is not None and tnP is np.array: # only peform check if explicitly specified
        if self.Pr != _Pr: raise ValueError("Row dimension of probability matrix incommensurate with input.")
        if self.Pc != _Pc: raise ValueError("Colum dimension of probability matrix incommensurate with input.")
      return

  def setP(self, _P = None, _x = None):  
    if _P is None: 
      self.B = False
      return self.setPdim()
    if isnum(_P):
      _P = tuple([_P])
    if type(_P) is tuple:
      return self.setB(_P)
    self.P = _P if type(_P) is not list else np.array(_P)

    if type(self.P) is np.ndarray:
      self.setPdim(list(self.P.shape))
      if self.pmat: self.setPmat(self.pmat)      # convert P from array->matrix
    elif type(self.P) is npmatrix:
      self.setPdim(self.P.shape)
      if not(self.pmat): self.setPmat(self.pmat) # convert P from matrix->array

    # Default x if not specified - this is outputted but does not here overwrite X
    if _x is None and self.NP:
      _x = [[]] * self.NP
      for i in range(self.NP):
        _x[i] = np.arange(self.nP[i], dtype = int)
      return _x
    return None
  def setX(self, _X = None):      
    if _X is None: 
      self.X = _X
      self.NX = 0
      self.nX = np.empty(0, dtype = int)
      return    
    if not(type(_X) is list): _X = [_X]
    self.NX = len(_X)    
    self.X = [[]] * self.NX  
    self.nX = np.ones( (self.NX), dtype = int)
    for i in range(self.NX):
      self.X[i] = np.array(_X[i], dtype = elType(_X[i]))
      self.nX[i] = len(self.X[i])
    # Default P dimensions if matrix specified
    if self.pmat and not(self.NP): self.setPdim(np.array(self.nX, dtype = int))
  def setB(self, _P = None): # sets boolean PMF
    if _P is None: return
    p = np.array(_P)
    n = len(p)
    self.setX([np.array((False, True), dtype = bool)]*n)
    mp = [[]] * n
    for i in range(n):
      mp[i] = np.array([1.-p[i], p[i]], dtype = float)
    self.setp(mp)
    return [np.array((False, True), dtype = bool)] * n
  def chk(self, Y, a = None):
    if a is None: a = range(self.NP)
    ys = Y.shape
    if self.pmat:
      if type(Y) is not npmatrix:
        raise TypeError("Mismatch in array/matrix data types.")
      if ys[0] != self.Pr and ys[1] != self.Pc:
        raise ValueError("Input incommensurate with the number of dimensions in probability mass.")   
      return
    if len(a) == self.NP:
      if len(ys) != self.NP:
        raise ValueError("Input incommensurate with the number of dimensions in probability mass.")   
    for i in a:
      if self.nP[i] != ys[i]:
        raise ValueError("Dimension " + str(i) + " of input incommensurate with probability mass.")
  def chkPX(self, chkX = False):
    if chkX:
      if self.X is None:
        raise ValueError("No vector value specifications set.")
    elif self.X is None: 
      return
    if self.P is None:
      raise ValueError("No probability mass array set.")
    if self.NP != self.NX:
      raise ValueError("Number of dimensions in probability mass inconsistent with vector value specifications.")
    for i in range(self.NP):
      if self.nP[i] != self.nX[i]:
        raise ValueError("Dimension " + str(i) + " in probability mass incommensurate with vector value specification.")
  def indr(self, a = 0, i = 0, _nR = None): # returns row indices for ith slice along axis a
    _nR = self.nR if _nR is None else np.array(_nR, dtype = int)
    if a == len(_nR): raise ValueError("indr() axis must be a non-column axis")
    return np.ravel(np.arange(np.prod(_nR), dtype=int).reshape(_nR).take(i, axis=a))
  def inda(self, a = 0): # returns row indices for values along axis a (or just columns)
    if a == self.NR: return np.arange(self.Pc, dtype = int)
    if len(self.rR) == 1: return np.array([0], dtype = int)
    ind = replicind(self.rR[a+1], self.nR[a], self.cR[a])
    return ind
  def setp(self, *_p): # this assumes independent marginal probabilities
    self.setPdim() # initialise P-dimensions
    _NP = 0
    pr = []
    if len(_p) == 1 and type(_p[0]) is list:
      _NP = len(_p[0])
      for i in range(_NP):
        pr.append(_p[0][i])
    else:
      for _p_ in _p:
        pr.append(_p_)
        _NP += 1
    _nP = [len(_pr) for _pr in pr]
    self.setPdim(_nP) # do not convert to _nP to array 
    for i in range(self.NP):
      if not(i):
        self.P = self.repa(i, pr[i])
      else:
        self.P = np.multiply(self.P, self.repa(i, pr[i]))
    if self.X is not None:
      self.chkPX(True)    
  def setx(self, _nx = 50, spec = 0, *_x):
    self.NX = 0;
    for x_ in _x:
      self.NX += 1
    if not(self.NX):
      _x = [np.array( (0.0, 1.0) )]
      self.NX = 1
    if isint(_nx): _nx = [_nx] * self.NX 
    self.nX = np.array((_nx), dtype = int)
    if not(self.nX.ndim): self.nX = self.nX.reshape((1))
    if type(spec) is int: spec = [spec] * self.NX
    self.X = [[]] * self.NX
    i = 0
    for x_ in _x:
      nx_ = len(x_)
      if nx_ != 2:
        if nx_ != self.nX[i]:
          raise ValueError("Each axis specification must consist of a two-element vector or limits or match size specification.")
        else:
          self.X[i] = x_
      else:
        self.X[i] = numspace(x_[0], x_[1], self.nX[i], spec[i])
      i += 1       
    if (not(self.P is None)):
      self.chkPX(True)       
  def repa(self, a = 0, x = None): # returns x[a] replicated for to the size and shape of self.P
    if x is None: x = self.X[a]
    if not(self.pmat):
      _NX, _nX = self.NX, self.nX
      if not(_NX):
        _NX = self.NP
        _nX = self.nP
      res = np.ones( (_NX), dtype = int)
      rep = np.array(_nX, dtype = int)
      if type(a) is int:
        res[a] = _nX[a]
        rep[a] = 1
      return np.tile(np.array(x).reshape(res), rep)        
    else:
      if a == self.NR:
        res = (1, self.Pc)
        rep = (self.Pr, 1)
        return np.tile(np.matrix(x).reshape(res), rep)        
      else:
        res = (self.Pr, 1)
        rep = (1, self.Pc)
        ind = self.inda(a)
        return np.tile(np.matrix(x[ind]).reshape(res), rep)        
  def margind(self, a = 0):       # return axis, and indices for a marginal operation (for arrays and matrices):
    if not(self.pmat): return a, None
    if a == self.NR: return 1, None
    return 0, inda(a)
  def calcCMF(self, C = None):
    if C is None:
      C = np.zeros(self.nP, dtype = int)
      o = np.ones(self.NP, dtype = int)
      for i in range(self.NP):
        d = np.arange(self.nP[i], dtype = int)
        res,    rep    = np.copy(o), np.copy(self.nP)
        res[i], rep[i] = self.nP[i], 1
        C += np.tile(d.reshape(res), rep)
    if type(C) is not list:
      c = C if C.ndim == 1 else C.ravel()
      m = C.max()
      M = m + 1
      C = [[]] * M
      for i in range(M):
        C[i] = np.nonzero(c == i)[0]
    pr = self.P.ravel()
    M = len(C)
    _X = np.arange(M, dtype = int)
    _P = np.empty(M, dtype = float)
    for i in range(M):
      c = C[i]
      if len(c):
        _P[i] = np.sum(pr[c])
    return mass(_P, _X, self.pmat)
  def copy(self):
    return mass(self.P.copy(), self.X[:], self.pmat)
  def setNormPrior(self, data = None, nx = 50, scale = 4):    
    mino = 1e-300
    if data is None:
      raise ValueError("Prior cannot be set without data.")
    if isint(scale): scale = [scale] * 2
    data = nanravel(data)
    nd = len(data)
    mn = nanmean(data)
    sd = nanstd(data)
    ms = nanste(data)*scale[0]
    ss = (1.0-nansdec(data))**scale[1] + mino
    self.setx(nx, [0, 1], [mn - ms, mn + ms], [(sd*ss)**2.0, (sd/ss)**2.0])
    self.setUniPrior()  
  def setUniPrior(self, spec = 0):
    mino = 1e-300
    if self.NX > 10:
      raise ValueError("setUniPrior() supports only up to 10 dimensions presently.")
    if isint(spec): spec = np.tile(spec, self.NX)
    pr = [[]] * self.NX
    for i in range(self.NX):
      if spec[i] == 0:
        _pr = np.ones(self.nX[i], dtype = float)
      elif spec[i] == 1:
        _pr = 1.0/(mino+np.fabs(self.X[i]))
      elif spec[i] == 2:
        _pr = 1.0/(mino+np.sqrt(np.fabs(self.X[i])))
      elif spec[i] == -1:
        _pr = 1.0 / (np.fabs(self.X[i]*(1.0 - self.X[i]))+mino)
      elif spec[i] == -2:
        _pr = 1.0 / (np.sqrt(np.fabs(self.X[i]*(1.0 - self.X[i])))+mino)
      pr[i] = _pr / (sum(_pr)+mino)  
    '''
    if self.NX == 1: 
      self.setp(pr[0])
    elif self.NX == 2: 
      self.setp(pr[0], pr[1])  
    elif self.NX == 3: 
      self.setp(pr[0], pr[1], pr[2])       
    elif self.NX == 4: 
      self.setp(pr[0], pr[1], pr[2], pr[3])        
    elif self.NX == 5: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4])
    elif self.NX == 6: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5])       
    elif self.NX == 7: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6])       
    elif self.NX == 8: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7])              
    elif self.NX == 9: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7], pr[8])       
    elif self.NX == 10: 
      self.setp(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7], pr[8], pr[9])
    '''
    self.setp(pr)
    self.normalise()
  def sample(self, _c):
    mino = 1e-300
    self.chkPX(True)
    if type(_c) is float: _c = np.array(_c, dtype = float)
    c = (np.arange(_c, dtype = float)+0.5)/(float(_c))*self.P.sum() if type(_c) is int else nanravel(_c)
    nc = len(c)
    _P = np.array(self.P).ravel()
    C = np.cumsum(_P)
    i = arghist(c, C)
    i[i < 0] = 0
    i[i > self.np] = self.np
    x = np.zeros( (nc, self.NP), dtype = float)
    if self.np == 1: # deal with exceptional case first
      for j in range(nc):
        for i in range(self.NP):
          x[j, i] = self.X[i][0]
      return x                          
    for j in range(nc):
      iinc = min(i[j]+1, self.np)
      ij = np.unravel_index(i[j], self.nP)
      for k in range(self.NX):
        ijk = ij[k]
        ijkinc = ijk + 1 if ijk < self.nX[k]-1 else ijk
        xijk = float(self.X[k][ijk])
        if ijk != ijkinc:
          p0 = _P[i[j]] + mino
          p1 = _P[iinc] + mino
          x[j][k] = (p0 * float(self.X[k][ijk]) + p1 * float(self.X[k][ijkinc]))/(p0 + p1)
    return x    
  def slice(self, a = 0, i = 0):
    sX = None
    if self.X is not None:
      sX = self.X[:]
      del sX[a]
    _P = self.P if self.pmat else np.matrix(self.P.reshape((self.Pr, self.Pc)))
    if a == self.NR:
      sP = _P[:, i]
      _Pr, _Pc = sP.shape[0], 1
      if self.NP > 1:
        _Pc = self.nP[-2]
        _Pr /= _Pc
        sP = sP.reshape((_Pr, _Pc))
    else:
      ri = self.indr(a, i)
      sP = _P[ri, :]
    if self.pmat:
      return mass(sP, sX, self.pmat)
    else:
      sPs = list(self.nP)
      del sPs[a]
      return mass(np.array(sP).reshape(sPs), sX, self.pmat)

  def setPslice(self, a, i, Pai):
    _P = self.P if self.pmat else np.matrix(self.P.reshape((self.Pr, self.Pc)))
    pai = Pai 
    if type(pai) is not npmatrix:
      pais = pai.shape
      if len(pais) < 2: 
        pai = np.matrix(pai.reshape((1, pais[0])))
      else:
        pai = np.matrix(pai.reshape((np.prod(pais[:-1]), pais[-1])))
    if a == self.NP:
      _P[:, i] = pai
    else:
      ri = self.indr(a, i)
      _P[ri, :] = pai
    '''
    if not(self.pmat):
      _P = _P.reshape(self.nP)
    '''
    self.setP(_P, True)
  def maxmassval(self):
    maxi = np.unravel_index(self.P.argmax(), self.nP)
    maxx = np.empty( (self.NX), dtype = float)
    for i in range(self.NX):
      maxx[i] = self.X[i][maxi[i]]
    return maxx   
  def normalise(self):
    mino = 1e-300
    self.P[self.P < 0.0] = 0.0
    self.P = self.P / (self.P.sum() + mino)
  def marginalise(self, _a = 0):
    if type(_a) is int:
      a = np.array(_a, dtype = int).reshape(1)
      na = 1
    else:
      a = np.unique(np.array(_a, dtype = int))[::-1] 
      na = len(a)
    MX = None
    if self.X is not None:
      MX = self.X[:]
      for i in range(na):
        del MX[a[i]]
    MP = np.array(self.P).reshape(self.nP) if self.pmat else self.P.copy() 
    for i in range(na):
      MP = MP.sum(axis = a[i])
    return mass(MP, MX, self.pmat)  
  def marginal(self, _a = 0):
    if not(type(_a) is int):
      raise ValueError("prob.marginal(axis): axis specification must be a single integer")
    a = np.arange(self.NP)
    a = np.delete(a, _a)
    return self.marginalise(a)
  def moment(self, _m = 0):
    self.chkPX(True)
    m = np.array(_m, dtype = int).reshape(1) if type(_m) is int else np.array(_m.ravel(), dtype = int)
    nm = len(m)
    x = np.zeros( (nm, self.NP), dtype = float)
    for i in range(self.NX):
      MP = self.marginal(i)
      p = MP.P
      for j in range(nm):
        mom = m[j]
        if not(mom):
          x[j,i] = p.sum()
        else:
          mom1 = (p*self.X[i]).sum()
          if mom == 1:
            x[j,i] = mom1
          else:
            x[j,i] = (p*(self.X[i] - mom1)**mom).sum()
    return x
  def conditionalise(self, _a = 0):
    mino = 1e-300
    if type(_a) is int:
      a = np.array(_a, dtype = int).reshape(1)
      na = 1
    else:
      a = np.unique(np.array(_a, dtype = int))[::-1] 
      na = len(a)
    C = np.array(self.P).reshape(self.nP) if self.pmat else self.P.copy()  
    for i in range(na):
      ai = a[i]
      resC = self.nX.copy()
      repC = np.ones(self.NX, dtype = int)
      resC[ai] = 1
      repC[ai] = self.nX[ai]
      C /= (np.tile(C.sum(axis = ai).reshape(resC), repC)+mino)
    return mass(C, self.X[:])
  def condition(self, other): 
    print("Warning: mass.condition is not type-safe")
    # self
    I0 = np.zeros(self.NX, dtype = int)
    for i in range(self.NX):
      _I0 = np.nonzero(self.X[i] == 0)[0]
      if len(_I0): I0[i] = _I0[0]
    I0 = tuple(I0)
    P0 = self.P[I0].sum()
    C = [P0, self.P.sum() - P0]
    # other
    I0 = np.zeros(other.NX, dtype = int)
    for i in range(other.NX):
      _I0 = np.nonzero(other.X[i] == 0)[0]
      if len(_I0): I0[i] = _I0[0]
    I0 = tuple(I0)
    P1 = other.P * C[1]
    P0 = np.zeros(other.nP, dtype = float)
    P0[I0] = C[0]
    # obj
    obj = mass()
    obj.setX(other.X)
    obj.setP(P0+P1)
    return obj
  def condmoment(self, _m = 0, _a = 0):
    if not(type(_m) is int):
      raise ValueError("Moment specification must be a single integer.")
    if not(type(_a) is int) or _a > 2: 
      raise ValueError("prob.marginal(axis): axis specification must be a single integer less than 2")
    a = _a
    m = _m
    d = self.nX[a]
    C = self.conditionalise(int(not(a)))
    sM = np.array( (d, C.NX - 1), dtype = int)
    M = np.empty( (sM), dtype = float)
    for i in range(d):
      M[i] = C.slice(a,i).moment(m)
    return M
  def multiply(self, _a = 0, postnormalise = False):
    mino = 1e-300
    _P = np.array(self.P).reshape(self.nP) if self.pmat else self.P
    if type(_a) is int:
      a = np.array(_a, dtype = int).reshape(1)
      na = 1
    else:
      a = np.unique(np.array(_a, dtype = int))[::-1] 
      na = len(a)
    LP = np.log(_P+mino)
    DL = 0.
    MX = self.X[:]
    for i in range(na):
      del MX[a[i]]
      lm = Max(LP, int(a[i]))
      DL += lm.max()
      LP -= reptile(lm, LP, a[i])
      LP = LP.sum(axis = a[i])
    _P = np.exp(LP)
    _P = _P / unzero(_P.sum()) if postnormalise else _P * np.exp(DL)
    return mass(_P, MX, self.pmat)   
  def spec2array(self, spec):
    if isint(spec): return self.repa(spec)
    if isfloat(spec): return np.array(spec, dtype = float).reshape(1)
    self.chk(spec)
    return spec
  def calcPost(self, lhpm, *args):
    # modifies current P (prior) by likelihood pmf lhpm using *args as value specification of lhpm for axes in self.X 
    mino = 1e-300
    if self.NX > 10:
      raise ValueError("calcPost() supports only up to 10 dimensions presently.")
    self.chkPX(True)
    lhpm.chkPX(True)    
    i = 0
    x = [[]] * len(args)
    for arg in args: 
      lhpm.chk(arg)
      x[i] = np.array(arg, dtype = float).ravel()
      i += 1
    if self.NP != i:
      raise ValueError("Number dimensions of probability distribution must match number of dependent variable inputs.")    
    I = np.empty((lhpm.np, self.NX), dtype = int)
    for i in range(self.NX):   
      Ii = arghist(x[i], self.X[i])
      Ii[Ii < 0] = 0
      Ii[Ii >= self.nX[i]] = self.nX[i] - 1 
      I[:,i] = Ii.copy()
    lhpmp = np.array(lhpm.P, dtype = float).ravel()    
    L = np.zeros(self.nP, dtype = float)
    for i in range(lhpm.np):
      Ii = I[i]
      if self.NX == 1:
        L[Ii] +=  lhpmp[i]   
      elif self.NX == 2:
        L[Ii[0], Ii[1]] += lhpmp[i]   
      elif self.NX == 3:
        L[Ii[0], Ii[1], Ii[2]] += lhpmp[i] 
      elif self.NX == 4:
        L[Ii[0], Ii[1], Ii[2], Ii[3]] += lhpmp[i] 
      elif self.NX == 5:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4]] += lhpmp[i]         
      elif self.NX == 6:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4], Ii[5]] += lhpmp[i]           
      elif self.NX == 7:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4], Ii[5], Ii[6]] += lhpmp[i]        
      elif self.NX == 8:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4], Ii[5], Ii[6], Ii[7]] += lhpmp[i]                
      elif self.NX == 9:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4], Ii[5], Ii[6], Ii[7], Ii[8]] += lhpmp[i]                        
      elif self.NX == 10:
        L[Ii[0], Ii[1], Ii[2], Ii[3], Ii[4], Ii[5], Ii[6], Ii[7], Ii[8], Ii[9]] += lhpmp[i]                                
    if self.pmat: L = np.matrix(L.reshape((self.Pr, self.Pc)))
    nc = self.P.sum() * lhpm.P.sum()  
    self.P = np.log(self.P + mino)+np.log(L + mino)
    self.P = np.exp(self.P - self.P.max())
    self.normalise()
    self.P *= nc    
  def calcNormPost(self, data = None, mom0 = 1.0, mom1 = 0, mom2 = 1, hidepb = False):
    # modifies current P (prior) by data to give normalised posterior of moment0, the Gaussian mean and variance
    mino = 1e-300
    if data is None:
      return mass(self.P.copy(), self.X[:])
    ndata = len(data)
    mom0 = self.spec2array(mom0)
    mom1 = self.spec2array(mom1)
    mom2 = self.spec2array(mom2)
    if len(mom0) == 1:
      ssd = -sumsqrdif(data, mom1, hidepb);
    else:
      ssd = np.zeros(mom0.shape, dtype = float)
      i = np.nonzero(mom0)
      ssd[i] = -sumsqrdif(data, mom1[i], hidepb);
    ssd /= (2.0*mom2 + mino)
    ssd += np.log(self.P+mino) + ndata * (np.log(mom0+mino) - 0.5 * np.log(2.0 * np.pi * mom2+mino))
    self.P = np.exp(ssd - ssd.max())
    self.normalise()
  def calcNormM1Post(self, data = None, mom1 = 0): 
    # modifies current P (prior) by data to give normalised posterior of Gaussian mean
    mino = 1e-300
    if data is None:
      return mass(self.P.copy(), self.X[:])
    ndata = len(data)
    mom1 = self.spec2array(mom1)
    ssd = sumsqrdif(data, mom1);
    ssd = np.log(self.P+mino) + (np.log(ssd+mino) * (0.5 * float(1 - ndata)))
    self.P = np.exp(ssd-ssd.max())
    self.normalise()    
  def calcMultNormPost(self, data = None, mom0 = 1.0, mom1 = 0, mom2 = 1, _a = 0, pb = None):    
    mino = 1e-300
    if type(_a) is int:
      a = np.array(_a, dtype = int).reshape(1)
      na = 1
    else:
      a = np.unique(np.array(_a, dtype = int))[::-1] 
      na = len(a)
    post = self.marginalise(a)
    if data is None:
      return post
    ndata = len(data)
    mom0 = self.spec2array(mom0)
    mom1 = self.spec2array(mom1)
    mom2 = self.spec2array(mom2)        
    m0ar = len(mom0) > 1
    posi = []
    l = None 
    if m0ar:
      posi = np.nonzero(mom0)
      if len(posi) == len(mom0.ravel()):
        m0ar = False
      else:
        l = np.zeros(mom0.shape, dtype=float)
    L = np.log(post.P)
    L -= L.max()
    C = np.log(mom0+mino) - 0.5 * np.log(mom2+mino)
    D = - 1.0 / (2.0 * mom2 + mino)      
    if not(pb is None): pb.setlast("Calculating...", ndata, 'b')
    for i in range(ndata):
      if not(pb is None): pb.updatelast(i)
      if m0ar:
        l[posi] = np.exp(C[posi] + D[posi] * ((data[i] - mom1[posi])**2.0))
      else:
        l = np.exp(C + D * ((data[i] - mom1)**2.0))
      if na == 1:
        L += np.log(l.sum(axis = a[0])+mino)
      else:  
        for j in range(na):
          l = l.sum(axis = a[j])
        L += np.log(sump+mino)
        if m0ar:
          l = np.zeros(mom0.shape, dtype=float)
    post.setP(np.exp(L - L.max()))
    post.normalise()
    return post
  
