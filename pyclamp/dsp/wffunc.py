# A waveform module

import numpy
import numpy as np
import scipy as sp
from pyclamp.dsp.sifunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.dtypes import *
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.optfunc import *

SHOWFITS = False
if SHOWFITS: 
  from mplqt import * 
  import iplot

def wavmatplot(X, samplint, colours = 'bgrcmy'):
  t = np.arange(X.shape[1], dtype = float) * samplint
  mn = np.mean(X, axis = 0)
  se = stderr(X, axis = 0)
  sd = np.std(X, axis = 0)
  q0 = np.min(X, axis = 0)
  q1 = np.percentile(X, 25, axis = 0)
  q2 = np.median(X, axis = 0)
  q3 = np.percentile(X, 75, axis = 0)
  q4 = np.max(X, axis = 0)
  mp.plot(t, mn, colours[0], label = 'Mean')
  mp.hold(True) 
  mp.plot(t, q2, colours[1], label = 'Median') 
  mp.plot(t, mn+se, colours[2], label = 'S.E.')
  mp.plot(t, mn-se, colours[2])
  mp.plot(t, mn+sd, colours[3], label = 'S.D.')
  mp.plot(t, mn-sd, colours[3])
  mp.plot(t, q1, colours[4], label = 'I.Q.R.')
  mp.plot(t, q3, colours[4])
  mp.plot(t, q0, colours[5], label = 'Range')
  mp.plot(t, q4, colours[5])
  return [ [t[0], t[-1]], [q0.min(), q4.max()] ]

def ezwavmatplot(X, samplint, colours = 'bc'):
  t = np.arange(X.shape[1], dtype = float) * samplint
  mn = np.mean(X, axis = 0)
  sd = np.std(X, axis = 0)
  lo = mn-sd
  hi = mn+sd
  mp.plot(t, mn, colours[0], label = 'Mean')
  mp.hold(True) 
  mp.plot(t, lo, colours[1], label = 'S.D.')
  mp.plot(t, hi, colours[1])
  return [ [t[0], t[-1]], [lo.min(), hi.max()] ]  

def offsetmean(X, spec = True, window = None):
  rX, cX = X.shape
  if window is None: window = cX
  if Type(window) is int: window = [0, window]
  if type(spec) is bool: spec = np.tile(spec, rX)
  if elType(spec) is bool: spec = argtrue(spec)
  Y = np.array(X, dtype = float)
  win = np.arange(window[0], window[1], dtype = int)
  if not(len(spec)) or not(len(win)): return np.copy(X)
  m = np.mean(Y[spec], axis = 0)[win]
  M = np.tile(m.reshape((1, len(m))), (rX, 1))
  Y[:, win] -= M
  return np.array(Y, dtype = elType(X))

def align(X, spec = 1, window = None): 
  if type(spec) is int:
    if window == None: window = [0, X.shape[1]]
    if window[0] == window[1]:
      return np.copy(X), [], []
    if len(window) == 2:
      window = [window[0], window[1], None]
    if spec == 1:
      i = X[:, window[0]:window[1]].argmax(axis=1)
      return align(X, i+window[0], window[2])
    elif spec == -1:
      i = X[:, window[0]:window[1]].argmin(axis=1)
      return align(X, i+window[0], window[2])
    else:
      return X[:, window[0]:window[1]], [], []
  if spec.ndim == 1 and spec.shape[0] == X.shape[0]:
    i = np.vstack([np.arange(X.shape[0]), spec])
    if window == None:
      window = [[]] * 2
      window[0] = -spec.min()
      window[1] = X.shape[1] - spec.max()
    if type(window) is tuple: # allow tweaked window size (debugging limits)
      window = list(window)
      window[0] = max(window[0], -spec.min())
      window[1] = min(window[1], X.shape[1] - spec.max())
      window = tuple(window)
    return align(X, i.T, window)
  if window == None:
    window = [[]] * 2
    window[0] = -spec[:,1].min()
    window[1] = X.shape[1] - spec[:,1].max()
  elif spec.ndim == 2:
    if type(window) is tuple: # allow tweaked window size (debugging limits)
      window = list(window)
      window[0] = max(window[0], -spec[:,1].min())
      window[1] = min(window[1], X.shape[1] - spec[:,1].max())
    else: # if windows is a list, it is enforced even if excluding data
      i = spec[:,1]
      ok = np.logical_and(i+window[0]>=0, i+window[1]<=X.shape[1])
      spec = spec[ok, :]
      window = [[]] * 2
      window[0] = -spec[:,1].min()
      window[1] = X.shape[1] - spec[:,1].max()
  else:
    raise ValueError("Unrecognised alignment specification format.")
  i = spec[:, 1]   
  ok = np.logical_and(i+window[0]>=0, i+window[1]<=X.shape[1])
  spec = spec[ok, :]
  wind = np.matrix(np.arange(window[0], window[1])) + np.matrix(spec)[:, 1]
  vind = np.tile(spec[:, 0].reshape(len(spec), 1), (1, wind.shape[1]))
  I = np.array(vind, dtype=int), np.array(wind, dtype=int)
  Y = X[I[0], I[1]]
  return Y, I, spec

def trigger(X, spec = 1, x = 0, d = 0):
  # spec can mean a variety of different things
  # +1: trigger on up   (where X>=x)
  # -1: trigger on down (where X<x)
  # array matching dimenions of x: trigger on delta x over delta spec
  # (where elements of spec are cumulative indices)

  if not(isarray(spec)):
    if spec == 1:        # trigger on up
      i = np.array(X >= x, dtype = int)
    elif spec == -1:     # trigger on down
      i = np.array(X <= x, dtype = int)
    else:
      raise ValueError("Unknown trigger specification: " + str(spec))
    ii = np.array(np.nonzero(diff0(i) == 1)).T
  else:                # trigger on gradients
    S = np.max(spec)
    n = len(spec)
    r, c = X.shape
    I = np.arange(S, c)
    C = c - S
    if C < 0: return np.zeros((0,0), dtype = int)
    O = np.ones((r, C), dtype = int)
    for h in range(n):
      s = spec[h]
      xh = x[h]
      if xh != 0.:
        z = 0 if not(h) else spec[h-1]
        lo, hi = np.arange(z, z+C), np.arange(s, s+C)
        D = X[:, hi] - X[:, lo]
        o = D >= xh if xh > 0. else D <= xh
        O = np.logical_and(O, o)
    i = np.zeros((r,c+1), dtype = int)
    i[:, 1:(C+1)] = O
    ii = np.array(np.nonzero(i[:,1:] - i[:,:-1] == 1)).T
  if d == 0: return ii
  done = len(ii) == 0
  j = 0
  while not(done):
    iij = ii[j]
    k = j + 1
    done = k == len(ii)
    if not(done):
      iik = ii[k]
      if iij[0] == iik[0] and iij[1] + d >= iik[1]:
        ii = np.delete(ii, k, 0)
      else:
        j += 1
  return ii     

    
def quiesce(i, Id = None, d = 0):
  n = len(i)
  if Id is None: Id = np.zeros(n, type = int)
  ok = np.ones(n, dtype = bool)
  done = n == 0
  j = 0
  k = 0
  while not(done):
    k += 1
    done = k == n
    if not(done):
      if Id[j] == Id[k] and i[j] + d >= i[k]:
        ok[k] = False
      else:
        j = k
  return ok


def translate(X, _x):
  x = np.matrix(_x.ravel())
  if isinstance(X, np.matrix): 
    return X - x.T
  else:
    return np.array(X - x.T)
  
def baseline(X, spec = None, mode = None, fixc = None):
  # spec can mean a variety things:
  # None                  : baseline to initial value
  # Integer               : baseline to X[:, 0:Integer]
  # [i0, i1]              : baseline to X[:, i0:i1]
  # [[i0, i1]]            : baseline to X[:, i0:i1]
  # [[i0, i1], [i2, i3]]] : baseline to X[:, i0:i1], X[:, i2:i3] etc...
  # 1D np.array           : baseline to array
  # 2D np.array           : baseline to array
  #
  # mode can mean three things (if spec is not an array):
  #
  # None                  : Same as false
  # False                 : Baseline to mean
  # True                  : Baseline to exponential fit
  # Positive integer      : Baseline to exponential fit of mode components
  # Negative integer      : Use derivate of separation -mode rather than baseline
  #
  # Input fixc 
  #
  # Integer/Float         : Baseline to exponential fit with this offset

  if nDim(X) == 1: X = np.array([X])
  N, n, dt = X.shape[0], X.shape[1], elType(X)
  if spec is None: spec = 1
  if isint(spec): spec = [0, spec]
  if type(spec) is not np.ndarray and nDim(spec) == 1: spec = [spec]
  if mode is None: mode = False
  if type(mode) is bool: mode = int(mode)
  c = fixc if fixc is None else float(fixc)

  if type(spec) is list:
    m = len(spec)
    J = [[]] * m
    J0 = np.inf
    J1 = 0
    for i in range(m):
      if len(spec[i]) != 2: raise ValueError("Unknown baseline specification: " + str(spec))
      j0, j1 = spec[i][0], spec[i][1]
      j0 = max(0, j0); j0 = min(j0, n)
      j1 = max(0, j1); j1 = min(j1, n)
      if j0 > j1: j0, j1 = j1, j0
      J0 = min(J0, j0)
      J1 = max(J1, j1)
      J[i] = [j0] if j1 - j0 < 2 else np.arange(j0, j1)
    j = np.hstack(J)
    if not(mode):
      if len(j) == 1:
        m = X[:, j]
      else:
        x = X[:, j]
        m = x.mean(axis = 1)
      return baseline(X, m)
    elif mode < 0:
      Y = np.copy(X)
      for i in range(m):
        Y[:, spec[i][0]:spec[i][1]] = diff0(Y[:, spec[i][0]:spec[i][1]], -mode)
      return Y
    else:
      x = np.array(j, dtype = float) - float(j.min())
      Y = np.array(X[:, j], dtype = float)
      k = np.arange(J0, n)
      f = np.array(k, dtype = float) - float(k.min())
      M = np.zeros((N, n), dtype = float)
      if c is not None:
        m = np.tile(c, n)
      if SHOWFITS:
        fig = iplot.figs()
      for i in range(N):
        y = Y[i,:]
        if c is None:
          P = expdfit(x, y, mode)[0]
          if np.isnan(P[0]):
            P[0] = y.mean()
            mk = np.tile(P[0], len(k))
          else:
            mk = expdval(P, f)
          p = np.minimum(mk.max(), np.maximum(P[0], mk.min()))
          m = np.tile(p, n)
          m[k] = mk
        else:
          p = exp0fit(x, y-c, mode)[0]
          if np.isnan(p[0]):
            mk = np.tile(y.mean(), len(k))
          else:
            mk = exp0val(p, f)+c
          m[k] = mk
        if SHOWFITS:
          fig.newPlot()
          plot(x, y, 'c.')
          hold(True)
          DJ = J1-J0
          plot(f[:DJ], mk[:DJ], 'k')
          title(str(i))
        M[i, :] = m
      return baseline(X, M)
  elif type(spec) is type(X):
    if nDim(spec) == 1:
      return np.array((np.array(X.T, dtype = float) - spec).T, dtype = dt)
    elif nDim(spec) == 2:
      return np.array((np.array(X, dtype = float) - spec), dtype = dt)
    else:
      raise ValueError("Unknown baseline specification: " + str(spec))
  raise ValueError("Unknown baseline specification: " + str(spec))
 
def minmaxwin(X, overall = False, win = None):
  rX, cX = X.shape
  if win is None: win = [0, cX]
  Xwin = X[:, win[0]:win[1]]
  if overall:
    x = np.mean(Xwin, axis=0)
    return X[:, np.argmin(x)+win[0]], X[:, np.argmax(x)+win[0]]
  else:
    return np.min(Xwin, axis = 1), np.max(Xwin, axis = 1)

def homogenise(_X, win = None, spec = None, overall = None): # match specific or overall minimum and/or maximum
  # spec == 1:   -> scale by maximum matching mean of previous maxima with new
  # spec == 0:   -> scale by range matching mean mid-point of previous with new
  # spec == -1:  -> scale by minimum matching mean of previous minima with new

  X = np.array(_X, dtype=float)
  rX, cX = X.shape

  if win is None: win = [0, cX]
  if spec is None: spec = 0
  if overall is None: overall = False

  xmin, xmax = minmaxwin(X, overall, win)
  mnxmin, mnxmax = xmin.mean(), xmax.mean()
  
  Xmin = np.tile(xmin.reshape((rX, 1)), (1, cX))
  Xmax = np.tile(xmax.reshape((rX, 1)), (1, cX))

  if spec == -1: # homogenise by minimum
    X *= mnxmin / unzero(Xmin)
    xmin, _ = minmaxwin(X, overall, win)
    Xmin = np.tile(xmin.reshape((rX, 1)), (1, cX))
    return np.array(X-Xmin+mnxmin, dtype = elType(_X))
  elif spec == +1: # homegenise by maximum
    X *= mnxmax / unzero(Xmax)
    _, xmax = minmaxwin(X, overall, win)
    Xmax = np.tile(xmax.reshape((rX, 1)), (1, cX))
    return np.array(X-Xmax+mnxmax, dtype = elType(_X))
  elif spec != 0:
    raise ValueError("Unknown specification: " + str(spec))

  xmid,   xwid   = 0.5 * (xmax + xmin)    , 0.5 * (xmax - xmin)
  mnxmid, mnxwid = 0.5 * (mnxmax + mnxmin), 0.5 * (mnxmax - mnxmin)

  Xmid = np.tile(xmid.reshape((rX, 1)), (1, cX))
  Xwid = np.tile(xwid.reshape((rX, 1)), (1, cX))

  X = mnxwid * (X - Xmid) / unzero(Xwid) + mnxmid
  return np.array(X, dtype = elType(_X))

def padReshape(x, nr, nc, padVal = None, padStart = 0):
  n = x.shape[0]
  N = nr * nc
  if not(n) or (not(N)):
    return np.empty(N)    
  if N < n:
    raise ValueError("padReshape: Input dimensions exceed dimensions for reshaped output")
  elif N == n:
    return x.reshape(nr, nc)
  if padVal == None:
    padVal = x[0] if padStart else x[-1]
      
  # need to pad
  d = N - n
  padx = np.tile(np.array(padVal), (1, d))
  if padStart:
    return np.append(padx, x).reshape(nr, nc)
  else:
    return np.append(x, padx).reshape(nr, nc) 

def argminfabs(x):
  i = x < 0
  x[i] = -x[i]
  return np.argmin(x)

def binomate(_x):
  x, n = _x, len(_x)
  m = n % 4
  n -= m
  if n < 1: return x[:0]
  x = _x[:n]
  r = int(n / 4)
  X = x.reshape((r, 4))
  x0, x1 = X.min(axis=1), X.max(axis=1)
  x0, x1 = x0.reshape((r, 1)), x1.reshape((r, 1))
  y = np.ravel(np.hstack((x0, x1)))
  if m:
    y[-2] = min(y[-2], _x[-m:].min())
    y[-1] = max(y[-1], _x[-m:].max())
  return y

class minmax:
  nm = None # minimum length before undertaking minmax
  nc = None # output length if minmax undertaken
  ni = None # input length
  no = None # output length
  bp = 0    # bool for bypassing minmax
  nRows = 0 # number of rows for pad reshaping
  nCols = 0 # number of columns for pad reshaping    
  def initialise(self, _nc = None, _nm = None, _ni = None):
    if _nc != None: 
      self.nc = _nc   
      if self.nc % 2:
        raise ValueError("minmax: for operational minmax, output length specification must be even")
    if _nm != None: 
      self.nm = _nm
    elif self.nc != None:
      self.nm = int(self.nc * np.log2(self.nc))
    if _ni != None: 
      self.ni = _ni
      if self.ni < self.nm:
        self.no = self.ni
        self.bp = 1
      else:
        self.no = self.nc 
        self.bp = 0
        self.nRows = int(self.nc / 2)           # number of points / 2 
        self.nCols = int(self.ni / self.nRows)  # number of columns most likely less one
        if self.nRows * self.nCols < self.ni:   # if the number of indices requires another column
          self.nCols += 1
  def setup(self, nCalcLength = None, nMinLength = None, i0 = None, i1 = None): # returns output length
    if i0 != None and i1 != None:
      _ni = i1 - i0 + 1
    else:
      _ni = None
    if self.nc != nCalcLength or self.nm != nMinLength or self.ni != _ni:
      self.initialise(nCalcLength, nMinLength, _ni)
    return self.no
  def calc(self, x, i0 = None, i1 = None):
    if i0 == None: i0 = 0
    if i1 == None: i1 = len(x)
    if self.ni != i1 - i0 + 1:
      self.setup(self.nc, self.nm, i0, i1)
    if self.bp:      
      return x[i0:i1+1]       
    X = padReshape(x[i0:i1+1], self.nRows, self.nCols)
    M = np.vstack( (X.min(1), X.max(1)) ).T
    return M.reshape(1, self.no) 
   
class minmax2: # performs decimation-in-time min-max operations on multi-episode single-channels
  defnd = 1920 # default number of pixels (nowadays use 2-4K, ideally 3072)
  defco = 2    # default relative overlap span
  def __init__(self, _dt = float):
    self.reset(_dt)
  def reset(self, _dt = float):
    self.dt = _dt                              # data-type of x and X
    self.ne = 0                                # number of episodes
    self.ns = 0                                # number of samples per episode
    self.nr = 0                                # number of running samples in last episode
    self.x = []                                # input data (list/array of np.arrays)
    self.o = []                                # latency index array
    self.X = [[np.array([], dtype = self.dt)]] # outer-most = by episode, middle = by decimation, inner = by sample
    self.nx = np.array([], dtype = int)        # number of min-maxed data
    self.lx = np.array([], dtype = int)        # entire numbers of data
    self.lX = [np.array([], dtype = int)]      # lengths of max-minned data
    self.overlay = False                       # boolean flag to overlay 
    self.T = []                                # output times
    self.Y = []                                # output values
  def setData(self, _x = [], _signof = 1., _o = []):
    if not(len(_x)): return
    if isnum(_signof): _signof = (_signof, 1, 0)
    self.si, self.gn, self.of = float(_signof[0]), float(_signof[1]), float(_signof[2])
    typex0 = type(_x[0])
    self.x = _x if typex0 is list or typex0 is np.ndarray else [_x]
    self.o = _o
    _ne = len(self.x)
    if self.ne != _ne: 
      self.newEpisode(_ne)
    _nr = len(self.x[self.ne-1])
    if self.nr != _nr: self.newData(_nr)
    if not(len(self.o)):
      if not(self.ns):
        self.o = np.zeros(1, dtype = int)
      else:
        self.o = np.array(np.arange(self.ne) * float(self.ns), dtype = int)
  def newEpisode(self, _ne = False):       # runs when we detect a new episode
    if _ne is False: _ne = len(self.x)
    if self.ne == 1:
      self.haveEpisode(len(self.x[0]))
    self.ne = _ne
    self.nx = np.hstack((self.nx, np.zeros(self.ne-len(self.nx), dtype = int)))       # concatenate record of max-minned counts
    self.lx = np.hstack((np.tile(len(self.x[0]), self.ne-1), len(self.x[self.ne-1]))) # concatenated array of dimensions of self.x
  def haveEpisode(self, _ns = False):      # runs when we have one completed episode
    if _ns is False: _ns = len(self.x[0])
    self.ns = _ns
  def newData(self, _nr = False):
    if _nr is False: _nr = len(self.x[self.ne[-1]])
    self.nr = _nr
    if not(self.ns): self.ns = self.nr
    newdata = self.nx < self.lx
    while len(self.X) < len(newdata):
      self.X.append([np.array([], dtype = self.dt)])
    while len(self.lX) < len(newdata):
      self.lX.append(np.array([], dtype = int))
    _newdata = False
    for i in range(self.ne):
      if newdata[i]:
        self.addMinMax(i)
        _newdata = True
    if _newdata:
      self.resetLast()
  def resetLast(self):      
    self.T = []
    self.Y = []
    self.II = [-1, -1]                         # record of last calculated index array
    self.cc = None                             # record of concatenation
    self.Ind = np.inf                          # record of last decimation order
  def addMinMax(self, i):
    xi = self.x[i]
    Xi = self.X[i]
    m = int(np.floor(np.log2(float(self.lx[i]))))
    while len(Xi) < m:
      Xi.append(np.array([], dtype = self.dt))
    while len(self.lX[i]) < m:
      self.lX[i] = np.hstack((self.lX[i], 0))
    for j in range(m):
      xij = Xi[j-1] if j else xi
      Xij = Xi[j] 
      self.X[i][j] = np.hstack((Xij, binomate(xij)))
      self.lX[i][j] = len(self.X[i][j]) 
    if len(self.lX[i]):
      if self.lX[i][m-1] == 0:
        self.lX[i] = self.lX[i][:-1]
        self.X[i] = self.X[i][:-1]
    self.nx[i] = len(xi) 
  def calcMinMax(self, _t0 = 0., _t1 = np.inf, _overlay = False, _nd = None, _co = None):
    if _nd is None: _nd = self.defnd
    if _co is None: _co = self.defco
    renew = self.overlay != _overlay
    self.overlay = _overlay
    if _overlay:
      renew = renew or self.calcOverlay(_t0, _t1, renew, _nd, _co)
    else:
      renew = renew or self.calcConcat(_t0, _t1, renew, _nd, _co)
    return renew
  def calcOverlay(self, _t0 = 0., _t1 = np.inf, renew = False, _nd = None, _co = None):
    if _nd is None: _nd = self.defnd
    if _co is None: _co = self.defco
    self.overlay = True
    nd, co = float(_nd), float(_co)
    mini, maxi = 0, self.ns
    mint, maxt = mini*self.si, maxi*self.si
    t0, t1 = min(_t0, _t1), max(_t0, _t1)
    t0, t1 = max(mint, t0), min(maxt, t1)
    i0, i1 = max(mini, int(t0/self.si)), min(maxi, int(t1/self.si))
    t0, t1 = float(i0) * self.si, float(i1) * self.si
    di = float(i1 - i0)
    if not(renew) and di <= 1.: return False # rubbish in, rubbish out
    nd, co = float(_nd), float(_co)
    ind = np.floor(np.log2(di / nd)) - 1.
    ind = int(max(-1, min(ind, len(self.lX[0])-1)))
    if not(renew):                                          # check if redraw needed
      if len(self.T) == self.ne and len(self.Y) == self.ne: # check if data present
        if ind >= self.Ind and ind < self.Ind + 2:          # detect large changes in detail level
          if i0 >= self.II[0] and i1 <= self.II[1]:         # check if outside window
            return False
    if len(self.Y) != self.ne: self.Y = [None] * self.ne
    if len(self.T) != self.ne: self.T = [None] * self.ne
    self.Ind = ind
    im = 0.5 * (float(i1+i0))
    iw = 0.5 * (float(i1-i0))
    i0 = int(max(0., im - iw * co))
    i1 = int(min(float(self.ns), im + iw * co))
    t0, t1 = float(i0) * self.si, float(i1) * self.si
    self.II = [i0, i1]
    I0, I1 = i0, i1
    div = 2.**float(ind+1.)
    if ind >= 0:
      I0, I1 = int(float(i0) / div), int(float(i1) / div)
    else:
      div = 1.
    mul = self.si * div
    for i in range(self.ne):
      y = self.X[i][ind] if ind >= 0 else self.x[i]
      self.Y[i] = np.array(y[I0:], dtype = float) if I1 > len(y) else np.array(y[I0:I1], dtype = float)
      if self.gn != 1.: self.Y[i] *= self.gn
      if self.of != 0.: self.Y[i] += self.of
      self.T[i] = t0 + mul * np.arange(len(self.Y[i]), dtype = float)
    return True  
  def calcConcat(self, _t0 = 0., _t1 = np.inf, renew = False, _nd = None, _co = None):
    if _nd is None: _nd = self.defnd
    if _co is None: _co = self.defco
    self.overlay = False
    nd, co = float(_nd), float(_co)
    mini, maxi = self.o[0], self.o[-1] + self.ns
    mint, maxt = mini*self.si, maxi*self.si
    t0, t1 = min(_t0, _t1), max(_t0, _t1)
    t0, t1 = max(mint, t0), min(maxt, t1)
    i0, i1 = max(mini, int(t0/self.si)), min(maxi, int(t1/self.si))
    t0, t1 = float(i0) * self.si, float(i1) * self.si
    di = float(i1 - i0)
    if di <= 1.: return False # rubbish in, rubbish out
    nd, co = float(_nd), float(_co)
    ind = np.floor(np.log2(di / nd)) - 1.
    ind = int(max(-1, min(ind, len(self.lX[0])-1)))
    ilo = np.nonzero(self.o < i0)[0]
    ihi = np.nonzero(self.o > i1)[0]
    ilo = ilo[-1] if len(ilo) else 0
    ihi = ihi[ 0] if len(ihi) else self.ne
    #print(mint, maxt, _t0, _t1, ind, ilo, ihi)
    if not(renew):                                          # check if redraw needed
      if len(self.T) == self.ne and len(self.Y) == self.ne: # check if data present
        if ind >= self.Ind and ind < self.Ind + 2:          # detect large changes in detail level
          if ilo >= self.II[0] and ihi <= self.II[1]:       # check if outside window
            return False
    if len(self.Y) != self.ne: self.Y = [None] * self.ne
    if len(self.T) != self.ne: self.T = [None] * self.ne
    self.Ind = ind
    im = 0.5 * (float(i1+i0))
    iw = 0.5 * (float(i1-i0))
    i0 = int(max(0., im - iw * co))
    i1 = int(min(float(maxi), im + iw * co))
    ilo = np.nonzero(self.o < i0)[0]
    ihi = np.nonzero(self.o > i1)[0]
    ilo = ilo[-1] if len(ilo) else 0
    ihi = ihi[ 0] if len(ihi) else self.ne
    self.II = [ilo, ihi]
    I = np.arange(self.ne, dtype = int)
    self.inrange = np.logical_and(I >= ilo, I < ihi)
    for i in list(I):
      if self.inrange[i]:
        self.Y[i] = np.array(self.X[i][ind], dtype = float) if ind >= 0 else np.array(self.x[i], dtype = float)
        self.T[i] = np.linspace(self.o[i], self.o[i]+self.lx[i], len(self.Y[i])) # always float
        if self.gn != 1.: self.Y[i] *= self.gn
        if self.of != 0.: self.Y[i] += self.of
        if self.si != 1.: self.T[i] *= self.si
      else:
        self.Y[i] = np.array((self.x[i][0], self.x[i][-1]), dtype = float)
        self.T[i] = np.array((self.o[i], self.o[i]+self.ns), dtype = float)
        if self.gn != 1.: self.Y[i] *= self.gn
        if self.of != 0.: self.Y[i] += self.of
        if self.si != 1.: self.T[i] *= self.si
    return True  
  def retXY(self): 
    return self.T, self.Y 
  def ind2X(self, _ind):
    ndind = nDim(_ind)
    if ndind == 1:
      ind = np.zeros(2, len(_ind), dtype = int)
      ind[1,:] = _ind
    elif ndind == 2:
      ind = _ind
    else:
      raise ValueError("Unknown index input array specification.")
    if self.overlay:
      return self.si * ind[1]
    return self.si * (self.o[ind[0]] + ind[1])
  def pick(self, _x, _y, boolok = True):
    if type(boolok) is bool: boolok = np.tile(boolok, self.ne)   
    boolko = np.logical_not(boolok)
    if not(boolok.sum()): return None
    if isarray(_x) and isarray(_y): # deal with range case first
      if len(_x) == 2 and len(_y) == 2:
        xr, yr = np.sort(_x), (np.sort(_y) - self.of) / unzero(self.gn)
        if not(self.overlay):
          midt = (np.array(self.o, dtype = float) + 0.5 * float(self.ns)) * self.si
          ok = np.logical_and(midt >= xr[0], midt <= xr[1])
        else:
          if self.ns == 0: return 0
          j0 = int(max(0, min(float(self.ns-1), round(np.floor(xr[0]/self.si)))))
          j1 = int(max(0, min(float(self.ns-1), round(np.ceil(xr[1]/self.si)))))
          j1 = max(j1, j0+1)
          xj = np.array(self.x)[:,j0:j1]
          xj[boolko, :] = MAXSI
          minx, maxx = xj.min(axis=1), xj.max(axis=1)
          ok = np.logical_and(maxx >= yr[0], minx <= yr[1])
        i = argtrue(np.logical_and(ok, boolok))
        if len(i): return i
      _x, _y = np.mean(_x), np.mean(_y)*self.gn + self.of
    if not(self.overlay): # easier - can ignore value of y
      midt = (np.array(self.o, dtype = float) + 0.5 * float(self.ns)) * self.si
      i = argtrue(boolok)
      return i[np.argmin(np.fabs(midt[i] - _x))]
    if self.ns == 0: return 0
    y_ =  (_y - self.of) / unzero(self.gn)
    j = int(max(0, min(float(self.ns-1), round(_x/self.si))))
    xj = np.array(self.x)[:,j]
    xj[boolko] = MAXSI
    return argminfabs(xj - y_)

def analyseInflection(w, _i0 = 0, _i1 = None, _polyn = 6, trimlim = np.inf):
  if _i1 == None: 
    _i1 = w.shape[0]
  polyn = _polyn - 1 
  polyn2 = polyn ** 2
  i0 = int(_i0)
  i1 = int(_i1)    
  if _i0 > _i1:
    i0 = _i1
    i1 = _i0
    
  # This next section is purely for the purposes of trimming  
  if i1 - i0 > trimlim:    
    imin = i0  
    imax = i1     
    i01 = np.arange(i0, i1)
    w01 = w[i01]   
    wmd = np.median(w01)
    wiq = np.percentile(w01, 75.0) - np.percentile(w01, 25.0)
    w01mxmd = np.max(w01) - wmd
    w01mnmd = wmd - np.min(w01)
    if w01mxmd > w01mnmd: # a peak
      ii = longest(w01 > wmd + wiq*1.5) 
    else: # a trough
      ii = longest(w01 < wmd - wiq*1.5)          
    if len(ii) > polyn2: # only trim if it is really worthwhile  
      i1 = ii[-1] + i0 #obviously needs to be this way round 
      i0 += ii[0]
      if i1 - i0 < polyn2:
        i1 += polyn2
        i0 -= polyn2
      if i1 > imax: i1 = imax
      if i0 < imin: i0 = imin   

  # Fit polynomial then solve roots of second derivative 
  x = np.arange(i0, i1, dtype = float)
  y = w[i0:i1]
  p = np.polyfit(x, y, polyn)  
  dp = np.polyder(p) 
  ddp = np.polyder(dp)
  r = np.roots(ddp)
  
  # For multiple roots solution, attempt to determine most relevant root
  r = np.real(r[np.nonzero(np.logical_not(np.iscomplex(r)))])
  if len(r) > 1:
    r = np.sort(r[np.nonzero(np.logical_and(r>=i0, r<i1))])
  i = 0
  if len(r) > 1:
    cc = np.corrcoef(np.arange(i0, i1), w[i0:i1])
    cc = cc[0,1]   
    hd = np.polyval(dp, r) 
    i = np.argmax(hd) if cc > 0 else np.argmin(hd) # select on basis whether rise and fall and then use the greatest
  ri = r[i]
  i = np.floor(ri) # do not add i0 since the offset x-values were used for fitting
  if i < i0: i = i0
  if i > i1: i = i1  
  y = np.polyval(p, i)
  d = np.polyval(dp, i)  
  return p, i, y, d

def argmaxdis(X, z = None, opts = 0):  # index of maximum displacement
  # opts: 0 = absolute, -1 = minimum, +1 = maximum
  nd = X.ndim
  if nd == 1: X = np.array([X])
  r, c = X.shape
  if z is None: z = np.zeros(r, dtype = float)
  D  = (X.T - z).T
  if opts < 0:
    D = -D
  elif opts == 0:
    D = np.fabs(D)
  s = D.sum(axis = 0)
  return np.argmax(s)

class maxminanal:
  def __init__(self, _polyn = 6):
    self.initialise(_polyn)
  def initialise(self, _polyn = 6):
    self.polyn = _polyn
  def analyse(self, W):
    self.analysemxmn(W)
    self.analyseinfl(W)
    self.sumstats()
  def analysemxmn(self, W):
    self.maxi = np.argmax(W, axis = 1)
    self.maxy = np.max(W, axis = 1)
    self.mini = np.argmin(W, axis = 1)
    self.miny = np.min(W, axis = 1) 
  def analyseinfl(self, W): 
    n = W.shape[0]
    m = W.shape[1]
    self.infi = np.empty( (n, 3), dtype = float)
    self.infy = np.empty( (n, 3), dtype = float)
    self.infd = np.empty( (n, 3), dtype = float)
    for i in range(n):
      w = W[i]
      if self.maxi[i] < self.mini[i]:
        p, self.infi[i][0], self.infy[i][0], self.infd[i][0] = analyseInflection(w, 0, self.maxi[i], self.polyn, self.polyn * 3)
        p, self.infi[i][1], self.infy[i][1], self.infd[i][1] = analyseInflection(w, self.maxi[i], self.mini[i], self.polyn, self.polyn * 3)
        p, self.infi[i][2], self.infy[i][2], self.infd[i][2] = analyseInflection(w, self.mini[i], m, self.polyn)
      else:
        p, self.infi[i][0], self.infy[i][0], self.infd[i][0] = analyseInflection(w, 0, self.mini[i], self.polyn, self.polyn * 3)
        p, self.infi[i][1], self.infy[i][1], self.infd[i][1] = analyseInflection(w, self.mini[i], self.maxi[i], self.polyn)
        p, self.infi[i][2], self.infy[i][2], self.infd[i][2] = analyseInflection(w, self.maxi[i], m, self.polyn)       
  def sumstats(self):
    self.maximn = np.mean(self.maxi, axis = 0)
    self.maxisd = np.std(self.maxi, axis = 0)
    self.maxymn = np.mean(self.maxy, axis = 0)
    self.maxysd = np.std(self.maxy, axis = 0)
    self.minimn = np.mean(self.mini, axis = 0)
    self.minisd = np.std(self.mini, axis = 0)
    self.minymn = np.mean(self.miny, axis = 0)
    self.minysd = np.std(self.miny, axis = 0) 
    finfi = np.array(self.infi, dtype = float)    
    self.infimn = np.mean(finfi, axis = 0)
    self.infisd = np.std(finfi, axis = 0)
    self.infymn = np.mean(self.infy, axis = 0)
    self.infysd = np.std(self.infy, axis = 0)
    self.infdmn = np.mean(self.infd, axis = 0)
    self.infdsd = np.std(self.infd, axis = 0)    
        
class fftfilter:  
  defaa = [-6.282258516862645, 0.0] # default  [attenuation, amplification] in nepers
  def __init__(self, _sr = 1., _aa = None):
    self.FIR = []
    self.X = None
    self.initialise(_sr, _aa)    
  def initialise(self, _sr = 1., _aa = None):
    self.setSamp(_sr)
    self.setAA(_aa)
    self.setFreq()
    self.setFilter()
  def setSamp(self, _sr = 1.):
    self.sr = _sr
    self.si = 1./self.sr
  def setAA(self, _aa = None):
    self.aa = self.defaa if _aa is None else _aa
    self.da = self.aa[1] - self.aa[0]
  def setFreq(self, _n = 0):
    eps = 1e-300
    self.n = _n
    self.f = None
    if self.n == 0: return
    self.odd = self.n % 2
    self.n1 = self.n-1 if self.odd else self.n
    self.n2 = int(self.n1 / 2)
    self.f = self.sr * (1.+np.arange(self.n1))/float(self.n1)   
    self.f2 = self.f[:self.n2]  
    self.F = np.log(self.f+eps)
    self.F2 = np.log(self.f2+eps)
    if self.N: self.setFilter()
  def addFIR(self, mode = None, freq = None, nlin = 0.): # mode: False = LPF, True = HPF
    if mode is None or freq is None: return
    if isnum(freq): 
      freq = np.tile(float(freq), 2)
    else:
      freq = np.array(freq, dtype = float)
    nlin = float(nlin)
    if mode is not None: 
      self.FIR.append([mode, freq, nlin])    
      self.N = len(self.FIR)
  def setFilter(self, _FIR = None):
    if _FIR is not None: self.FIR = _FIR
    self.fm = []
    self.N = 0
    if self.n == 0: return self.fm
    if self.f is None: self.setFreq(self.n)
    self.N = len(self.FIR)
    self.filters = np.empty((self.N, self.n1), dtype = float)
    for i in range(self.N):
      self.filters[i,:] = self.calcFIR(i)
    self.Filter = np.prod(self.filters, axis = 0)  
    self.fm = np.exp(self.aa[0] + self.da * self.Filter)
    return self.fm
  def calcFIR(self, _fir = None):
    eps = 1e-300
    if _fir is None or self.n is None: return None    
    if isint(_fir): _fir = self.FIR[_fir]
    if self.f is None: self.setFreq(self.n)
    mode, frq0, frq1, nlin = _fir[0], float(_fir[1][0]), float(_fir[1][1]), float(_fir[2])
    if mode is None: return np.ones(self.n, dtype = float)
    fl, fh = min(frq0, frq1), max(frq0, frq1)
    ll, lh = np.log(fl+eps), np.log(fh+eps)
    mf = np.zeros(self.n2, dtype = float)
    i = np.nonzero(np.logical_and(self.f2 >= fl, self.f2 <= fh))[0]
    if mode: # HPF
      mf[self.f2 > fh] = 1.
    else:    # LPF
      mf[self.f2 < fl] = 1.
    ni = len(i)        
    if ni == 1: # handle trivial clase
      mf[i] = 0.5
    elif ni > 1:  
      lm = 0.5 * (lh + ll)
      lw = 0.5 * (lh - ll)
      lf = self.F2[i]
      ln = (lf - lm) / (lw+eps)
      if nlin < eps:
        if not(mode): ln = -ln
        nl = ln
      else:
        if mode: # HPF
          ln *= nlin
        else:    # LPF
          ln *= -nlin
        nl = np.tanh(ln)
      mn, mx = nl.min(), nl.max()
      mf[i] = (nl - mn) / (mx - mn + eps)
    return np.hstack((mf, mf[::-1]))
  def setData(self, _X = None, _sr = None):
    self.m = 0
    if _sr is not None: self.setSamp(_sr)
    if _X is None: return
    self.nd = nDim(_X)
    if self.nd == 1: _X = np.array([_X])
    self.m, _n = _X.shape[0], _X.shape[1]
    if _n % 2:
      self.X = _X[:, :-1]
    else:
      self.X = _X
    if self.n != _n or _sr is not None: self.setFreq(_n)
  def process(self, _X = None, _sr = None, pgb = None):
    if _X is not None: self.setData(_X, _sr)
    if self.X is None: return None
    N1 = len(self.X)
    N2 = N1 * 2
    if pgb is not None: pgb.init("Filtering data", N2)
    _fX = [[]] * N1
    k = -1
    for i in range(N1):
      k += 1
      if pgb is not None: pgb.set(k)
      _fX[i] = np.fft.fft(self.X[i])
    self.fX = np.array(_fX)
    #self.fX = np.fft.fft(self.X)
    self.fY = self.fX * self.fm
    #self.Y = np.real(np.fft.ifft(self.fY))
    _Y = [[]] * N1
    for i in range(N1):
      k += 1
      if pgb is not None: pgb.set(k)
      _Y[i] = np.real(np.fft.ifft(self.fY[i]))
    if pgb is not None: pgb.close()
    self.Y = np.array(_Y)
    if self.odd: self.Y = np.hstack((self.Y, self.Y[:,-1].reshape(self.m, 1)))
    if self.nd == 1: return self.Y[0]
    return self.Y
  
def lpfilter(X, sr = 1., fr = None, nl = 0., pgb = None, opts = False): # opts = True returns class aswell
  dt = elType(X)
  if fr is None: fr = 0.5*sr
  if isnum(fr): fr = [fr, fr]
  ff = fftfilter(sr)
  ff.addFIR(False, fr, nl)
  Y = np.array(ff.process(X, sr, pgb), dtype = dt)
  if not(opts): return Y
  return Y, ff

def hpfilter(X, sr = 1., fr = None, nl = 0., pgb = None, opts = False): # opts = True returns class aswell
  dt = elType(X)
  if fr is None: fr = 2.*sr / float(np.max(X.shape))
  if isnum(fr): fr = [fr, fr]
  ff = fftfilter(sr)
  ff.addFIR(True, fr, nl)
  Y = np.array(ff.process(X, sr, pgb), dtype = dt)
  if not(opts): return Y
  return Y, ff

def bnfilter(X, n):
  d1 = np.ndim(X) == 1
  Y = np.array([X], dtype = float) if d1 else np.array(X, dtype = float)
  N = len(Y)
  c = sp.misc.comb(n, np.arange(n))
  c = c / c.sum()
  for i in range(N):
    Y[i] = np.convolve(Y[i], c, mode='same')
  return Y

