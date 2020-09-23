# NQS class simulators

import numpy as np
import scipy as sp
import scipy.stats as stats
import pyclamp.dsp.discprob
from pyclamp.qnp.qmod import *
from pyclamp.dsp.optfunc import *
from pyclamp.dsp.fpfunc import *

def amp2wave(_A, si = 0.001, e = 0.05, T = [-2., 8.], ldc = [2., 1.], sn = None):
  nd = nDim(_A)
  A = _A if nd == 2 else [_A]
  if sn is None: sn = np.sign(np.mean(np.ravel(A)))
  N = len(A)
  t = np.arange(T[0], T[1], si)
  m = len(t)
  I = t > 0
  W = [[]] * N
  p = np.array([0., 1., ldc[0], 1., ldc[1]])
  u = np.fabs(exppval(p, t[I]))
  u /= unzero(u.max())
  for i in range(N):
    a = A[i]
    n = len(a)
    w = np.empty((n, m), dtype = float)
    for j in range(n):
      w[j] = np.random.normal(0., e, m)
      w[j][I] += u * a[j]
    W[i] = w.copy()
  if nd < 2: return W[0]
  return W
  
class qus:
  def __init__(self, _q = 1.0, _u = 0.0, _s = 0.5, _usegamma = False):
    self.setqus(_q, _u, _s, _usegamma)  
  def setqus(self, _q = None, _u = None, _s = None, _usegamma = False):    
    mino = 1e-300
    if not(_q is None): self.q = float(_q);
    if not(_u is None): self.u = float(_u); 
    if not(_s is None): self.s = float(_s);
    self.usegamma = _usegamma
    self.qu = self.q * self.u
    self.qu2 = self.qu**2.0
    if self.usegamma:
      if self.u > 1.0:
        raise ValueError('Intra-site c.v. cannot exceed one for a gamma distribution')      
      pol = 1.0 if self.q > 0.0 else -1.0
      self.g = 1.0 / (self.u ** 2.0 + mino)
      self.l = self.q / (self.g + mino)
      
  def simulate(self, num = 1):
    mino = 1e-300
    if not(self.usegamma):
      if self.qu < mino:
        _Q = np.tile(self.q, num)
      else:  
        _Q = np.random.normal(self.q, self.qu, num);
    else:
      _Q = np.sign(self.q) * np.random.gamma(self.g, np.fabs(self.l), num) 
    _S = np.random.binomial(1, self.s, num)
    return _Q * _S

class enquvs:
  def __init__(self, _e = 0.0, _n = 1, _q = 1.0, _u = 0.0, _v = 0.0, _s = 0.5, _usegamma = [False, False]): #usegamma = [Intrasite, Intersite] gamma
    self.setnoise(_e)
    self.setnquvs(_n, _q, _u, _v, _s, _usegamma)
  def setnoise(self, _e = None):
    if isint(_e): _s = float(_e)
    if not(_e is None): self.e = _e;
  def setnquvs(self, _n = None, _q = None, _u = None, _v = None, _s = None, _usegamma = [False, False]):
    mino = 1e-300
    minl = 1e-3    
    if isfloat(_n): _n = int(_n)
    if isint(_q): _q = float(_q)
    if isint(_u): _u = float(_u)
    if isint(_v): _v = float(_v)
    if isint(_s): _s = float(_s)
    if not(_n is None): self.n = _n;
    if not(_q is None): self.q = _q;
    if not(_u is None): self.u = _u;  
    if not(_v is None): self.v = _v;  
    if not(_s is None): self.s = _s;  
    self.usegamma = _usegamma
    if not(isfloat(self.v)):
      raise ValueError('Inter-site c.v. must be a single value')
    if not(isint(self.n)):
      raise ValueError('Release site number must be a single value')    
    self.QUS = [[]];
    if self.n < 1: return
    self.QUS = [[]] * self.n    
    if isfloat(self.q):
      self.qv = self.q * self.v
      c = (np.arange(self.n, dtype = float)+0.5)/(float(self.n))
      if not(_usegamma[1]):
        _Q = stats.norm.ppf(c, np.fabs(self.q), np.fabs(self.qv+mino))
        _Q = (_Q - np.mean(_Q)) * np.fabs(self.qv) / (np.std(_Q) + mino) + np.fabs(self.q)
      else:
        if np.fabs(self.v) > np.fabs(self.q):
          raise ValueError('Inter-site c.v. cannot exceed one for a usegamma distribution')
        _g = 1.0 / (self.v ** 2.0 + mino)
        _l = max(minl, np.fabs(self.q / (_g + mino)))
        if self.v < mino:
          _Q = np.tile(self.q, self.n)
        else:  
          _Q = stats.gamma.ppf(c, _g, 0.0, _l)
        done = False  
        while not(done):
          _Q = (_Q - np.mean(_Q)) * (np.fabs(self.qv)+mino) / (np.std(_Q) + mino) + np.fabs(self.q)
          if _Q.min() >= mino:
            done = True
          else:  
            _Q[_Q < mino] = mino
      _Q *= np.sign(self.q)
    for i in range (self.n):
      if isfloat(self.q):        
        _q = _Q[i]
      else:
        _q = self.q[i]
      _u = self.u if isfloat(self.u) else self.u[i]
      _s = self.s if isfloat(self.s) else self.s[i]
      self.QUS[i] = qus(_q, _u, _s, _usegamma[0])
  def simulate(self, num = 1):
    _NQS = np.random.normal(0.0, self.e, num)
    for _qvs in self.QVS:
      _NQS += _qus.simulate(num)
    return _NQS;

class binoqsim (enquvs):
  def __init__(self, _e = 0.0, _n = 1, _q = 1.0, _u = 0.0, _v = 0.0, _s = 0.5, _usegamma = [False, False]): 
    self.setnoise(_e)
    self.setnquvs(_n, _q, _u, _v, _s, _usegamma)  
  def simulate(self, num = 1, _S = None):
    mino = 1e-300
    if not(isfloat(self.q)) or not(isfloat(self.u)):
      raise ValueError("binoqsim: quantal size and inter-site variance must each be a single floating point value and not an array.")
    S = _S
    if S is None:
      S = np.array(self.S, dtype = float) if Type(self.s) is float else self.s
    elif type(S) is list:
      S = np.array(_S, dtype = float)
    _NQS = np.empty( (S.shape[0], num), dtype = float )
    re = self.e + mino
    for i in range(S.shape[0]):
      if not(_S is None): self.setnquvs(self.n, self.q, self.u, self.v, S[i], self.usegamma)
      _NQS[i] = np.random.normal(0.0, re, num)       
      for _qus in self.QUS:
        _NQS[i] += _qus.simulate(num)
    return _NQS  
  
class betaqsim (binoqsim):
  def __init__(self, _e = 0.0, _n = 1, _q = 1.0, _u = 0.0, _v = 0.0, _a = 0.5, _s = 0.5, _usegamma = [False, False]): 
    mino = 1e-300
    if _v > mino:
      raise ValueError('Inter-site variance must be set to zero for non-uniform probability simulation.')
    self.setnoise(_e)
    self.setalpha(_a)
    self.setnquvs(_n, _q, _u, _v, _s, _usegamma)
  def setalpha(self, _a = 0.5):
    self.a = _a
  def simulate(self, num = 1, _S = None):
    mino = 1e-300
    if not(isfloat(self.q)) or not(isfloat(self.u)):
      raise ValueError("betaqsim: quantal size and inter-site variance must each be a single floating point value and not an array.")
    S = _S
    if S is None:
      S = np.array(self.S, dtype = float) if Type(self.s) is float else self.s
    _NQS = np.empty( (S.shape[0], num), dtype = float )
    re = self.e + mino   
    self.Si = np.tile(S.reshape((S.shape[0], 1)), [1, self.n])
    for i in range(S.shape[0]):      
      if not(_S is None):
        c = (np.arange(self.n, dtype = float)+0.5)/(float(self.n))
        b = self.a * (1.0 / (S[i]) - 1.0)
        si = discprob.betasample(self.n, self.a, b, True)       
        si.sort()  
        self.Si[i, :] = si.copy()
        self.setnquvs(self.n, self.q, self.u, self.v, si, self.usegamma)
      _NQS[i] = np.random.normal(0.0, re, num)       
      for _qus in self.QUS:
        _NQS[i] += _qus.simulate(num)
    return _NQS    
  
