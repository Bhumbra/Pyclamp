import numpy as np
import warnings
from lsfunc import *
from dtypes import *
from cstats import *
from sifunc import *
from numpy.core.umath_tests import inner1d
import scipy.signal as signal
import scipy.special as special
import scipy.stats as stats

npmatrix=np.matrixlib.defmatrix.matrix
__SI_MAX__ = 9223372036854775807
__FP_MAX__ = 1.7976931348623157e308

#import warning

#-----------------------------------------------------------------------------------------------------------------------
def list2nan2D(X):
  n = len(X)
  N = 0
  for i in range(n):
    N = max(N, len(X[i]))
  Y = np.tile(np.nan, (n, N))
  for i in range(n):
    x = nanravel(np.array(X[i], dtype = float))
    Y[i][:len(x)] = np.copy(x)
  return Y

#-----------------------------------------------------------------------------------------------------------------------
def diag_inner(_A, _B, T = True):
  #'''
  if type(_A) is np.matrix:
    C = np.mat(np.einsum('ij,ij -> i', _A, _B))
    return C.T
  else:
    C = inner1d(_A, _B)
    return C
  #'''
  '''
  A, B = np.array(_A), np.array(_B)
  Ar, _ =  A.shape

  C = np.matrix(np.empty((Ar, 1), dtype = float)) if T else np.matrix(np.empty((1, Ar), dtype = float))
  if T:
    for i in range(Ar):
      C[i] = np.vdot(A[i], B[i])
  else:
    for i in range(Ar):
      C[0, i] = np.vdot(A[i, :], B[i, :]).T

  return np.array(C)
  '''

#-----------------------------------------------------------------------------------------------------------------------
def fibonacci(n): # int(n) means return first n, whereas float(n) means limit
  j = 1
  k = 0
  if Type(n) is int:
    K = np.zeros(n, dtype = int)
    for i in range(1,n):
      k, j = k+j, k
      K[i] = k
    return K
  m = int(n)
  # handle special cases first
  if m == 0: return np.array([], dtype = int)
  if m == 1: return np.array([0,1,1], dtype = int)
  if m == 2: return np.array([0,1,1,2], dtype = int)
  if m < 5: return np.array([0,1,1,2,3], dtype = int)
  if m < 8: return np.array([0,1,1,2,3,5], dtype = int)
  K = np.zeros(m, dtype = int)
  h = 0
  i = 0
  while k < m:
    i += 1
    k, j = k+j, k
    if k <= m:
      K[i] = k
      h = i+1
  return K[:h]

#-----------------------------------------------------------------------------------------------------------------------
def arghist(x, bins):
  if not(len(x)): return np.array([], dtype = int)
  if not(len(bins)): return None
  y = np.digitize(x.ravel(), bins.ravel())
  if bins[-1] >= bins[0]: y -=  1
  return nanravel(y)

#-----------------------------------------------------------------------------------------------------------------------
def freqhist(x, bins):
  return np.hstack( (np.histogram(x, bins)[0], 0) )

#-----------------------------------------------------------------------------------------------------------------------
def digitise(x, bins):
  return arghist(x, bins)

#-----------------------------------------------------------------------------------------------------------------------
def iszero(x, eps = 1e-300):
  return np.fabs(x) < np.fabs(eps)

#-----------------------------------------------------------------------------------------------------------------------
def isone(x, _sig = 1e-15):
  sig = np.fabs(_sig)
  return np.logical_and(1.-sig < x, x < 1.+sig)

#-----------------------------------------------------------------------------------------------------------------------
def unzero(x, eps = 1e-300):
  if x is None: return None
  y = np.array(x, dtype = float)
  z = iszero(x, eps)
  if np.any(z):
    if y.ndim == 0:
      y = eps
    else:
      y[z] = eps
  return y

#-----------------------------------------------------------------------------------------------------------------------
def isequal(x, y, relative = False):
  if isint(x): x = float(x)
  if isint(y): y = float(y)
  if not(relative):
    return iszero(np.fabs(x-y))
  return np.logical_or(isone(x/unzero(y)), np.logical_and(iszero(x), iszero(y)))

#-----------------------------------------------------------------------------------------------------------------------
def atan2(x, y):
  return np.arctan2(y, x)

#-----------------------------------------------------------------------------------------------------------------------
def cart2pol(x, y):
  z = np.sqrt(x**2.+y**2.)
  w = atan2(x, y)
  return [w, z]

#-----------------------------------------------------------------------------------------------------------------------
def pol2cart(w, z):
  x = z * np.cos(w)
  y = z * np.sin(w)
  return [x, y]

#-----------------------------------------------------------------------------------------------------------------------
def cart2ellipse(_x, _y, _xyps = []): # (x0, y0, phase, scale)
  xyps = listcomplete(_xyps, [0., 0., 0., 1.])
  [w, z] = cart2pol(_x - xyps[0], _y - xyps[1])
  [x, y] = pol2cart(w - xyps[2], z)
  return cart2pol(x, y / unzero(xyps[3]))

#-----------------------------------------------------------------------------------------------------------------------
def ellipse2cart(_w, _z, _xyps = []): # (x0, y0, phase, scale)
  xyps = listcomplete(_xyps, [0., 0., 0., 1.])
  [x, y] = pol2cart(_w, _z)
  [w, z] = cart2pol(x, y * xyps[3])
  [x, y] = pol2cart(w + xyps[2], z)
  return [x + xyps[0], y + xyps[1]]

#-----------------------------------------------------------------------------------------------------------------------
def cart2normel(_x, _y, _xyabps = []): # (x0, y0, xnorm, ynorm, phase, scale)
  xyabps = listcomplete(_xyabps, [0., 0., 1., 1., 0., 1.])
  [w, z] = cart2pol((_x - xyabps[0])/unzero(xyabps[2]), (_y - xyabps[1])/unzero(xyabps[3]))
  [x, y] = pol2cart(w - xyabps[4], z)
  return cart2pol(x, y / unzero(xyabps[5]))

#-----------------------------------------------------------------------------------------------------------------------
def normel2cart(_w, _z, _xyabps = []): # (x0, y0, xnorm, ynorm, phase, scale)
  xyabps = listcomplete(_xyabps, [0., 0., 1., 1., 0., 1.])
  [x, y] = pol2cart(_w, _z)
  [w, z] = cart2pol(x, y * xyabps[5])
  [x, y] = pol2cart(w + xyabps[4], z)
  return [x*xyabps[2] + xyabps[0], y*xyabps[3] + xyabps[1]]

#-----------------------------------------------------------------------------------------------------------------------
def diff0(X, d = 1, ax = 0):
  nd = X.ndim
  if nd < 1 or nd > 2: raise ValueError("diff0: supports only 1 or 2 dimensions")
  et = elType(X)
  fd = float(d)
  _d2 = int(round(0.5*fd))
  d2_ = int(d - _d2)
  if nd == 1:
    if ax != 0: print("diff0 Warning: axis specification invalid for 1-d array")
    Y = (X[d:] - X[:-d]) / fd
    _z2 = np.zeros(_d2, dtype = et)
    z2_ = np.zeros(d2_, dtype = et)
    return np.hstack( (_z2, Y, z2_) )
  if ax == 0:
    Y = (X[:,d:] - X[:,:-d]) / fd
    n = Y.shape[0]
    _z2 = np.zeros((n, _d2), dtype = et)
    z2_ = np.zeros((n, d2_), dtype = et)
    return np.hstack( (_z2, Y, z2_) )
  Y = (X[d:,:] - X[:-d,:]) / fd
  n = Y.shape[1]
  _z2 = np.zeros((_d2, n), dtype = et)
  z2_ = np.zeros((d2_, n), dtype = et)
  return np.vstack( (_z2, Y, z2_) )
#-----------------------------------------------------------------------------------------------------------------------
def nanravel(X):
  if type(X) is float or type(X) is np.float64:
    X = np.array(X, dtype = float)
  x = X.ravel()
  if not(x.ndim):
    x = x.reshape(1)
  x = x[np.nonzero(np.logical_not(isNaN(x)))]
  if x.ndim:
    return x
  return x.reshape(1)

#-----------------------------------------------------------------------------------------------------------------------
def nan2val(X, val = 0.0):
  if not(isarray(X)):
    if isnumeric(X):
      Y = float(X)
      if not(np.isnan(Y)):
        return Y
    return val
  Y = np.array(X, dtype = float)
  i = isNaN(Y)
  Y[i] = val
  return Y

#-----------------------------------------------------------------------------------------------------------------------
def nanmedian(X):
  return np.median(nanravel(X))

#-----------------------------------------------------------------------------------------------------------------------
def nansign(X):
  Y = nan2val(X, 0)
  return np.array(np.sign(Y), dtype = float)

#-----------------------------------------------------------------------------------------------------------------------
def nancnt(X, **kwArgs):
  return np.sum(np.array(np.logical_not(isNaN(X)), dtype = int), **kwArgs)

#-----------------------------------------------------------------------------------------------------------------------
def nandem(X, **kwArgs):
  onem = 1e-16 # minimum number to be preserved within FP mantissa
  n = np.array(nancnt(X, **kwArgs), dtype = int)
  n += np.array(np.logical_not(n), dtype = int)
  return np.array(n, dtype = float) + onem

#-----------------------------------------------------------------------------------------------------------------------
def nansum(X, **kwArgs):
  return np.sum(nan2val(X), **kwArgs)

#-----------------------------------------------------------------------------------------------------------------------
def nanmean(X, **kwArgs):
  return nansum(X, **kwArgs) / nandem(X, **kwArgs)

#-----------------------------------------------------------------------------------------------------------------------
def nanmoment(X, M = 0, **kwArgs):
  return nanmean(X**float(M), **kwArgs)

#-----------------------------------------------------------------------------------------------------------------------
def nanmom2(X, **kwArgs):
  return (nanmoment(X, 2, **kwArgs) - nanmean(X, **kwArgs)**2.0)

#-----------------------------------------------------------------------------------------------------------------------
def nanmom3(X, **kwArgs):
  m1 = nanmean(X, **kwArgs)
  m2 = nanmoment(X, 2, **kwArgs)
  return (nanmoment(X, 3, **kwArgs) - 3.0*m1*m2 + 2.0*m1**3.0)

#-----------------------------------------------------------------------------------------------------------------------
def nanmom4(X, **kwArgs):
  m1 = nanmean(X, **kwArgs)
  m2 = nanmoment(X, 2, **kwArgs)
  m3 = nanmoment(X, 3, **kwArgs)
  return nanmoment(X, 4, **kwArgs) - 4.0*m1*m3 + 6.0*m2*m1**2.0 - 3.0*m1**4.0

#-----------------------------------------------------------------------------------------------------------------------
def nanvar(X, **kwArgs):
  n = nandem(X, **kwArgs)
  return nanmom2(X, **kwArgs) * n / (n - 1.0)

#-----------------------------------------------------------------------------------------------------------------------
def nanskew(X, **kwArgs):
  mino = 1e-300
  return nanmom3(X, **kwArgs) / (nanmom2(X, **kwArgs)**1.5+mino)

#-----------------------------------------------------------------------------------------------------------------------
def nankurt(X, **kwArgs):
  mino = 1e-300
  return nanmom4(X, **kwArgs) / (nanmom2(X, **kwArgs)**2.0+mino)

#-----------------------------------------------------------------------------------------------------------------------
def nanstd(X, **kwArgs):
  return np.sqrt(nanvar(X, **kwArgs))

#-----------------------------------------------------------------------------------------------------------------------
def nanste(X, **kwArgs):
  return nanstd(X, **kwArgs) / np.sqrt(nandem(X, **kwArgs))

#-----------------------------------------------------------------------------------------------------------------------
def nansdec(X, **kwArgs):
  n = nandem(X, **kwArgs)
  nd = n - 1.0
  ndh = 0.5 * nd;
  G = special.gammaln(0.5*n) - special.gammaln(ndh)
  K = np.sqrt(ndh) * np.exp(-G)
  V = 2.0 * (ndh - np.exp(2.0*G))
  return K * np.sqrt(V) / np.sqrt(nd)

#-----------------------------------------------------------------------------------------------------------------------
def nansde(X, **kwArgs):
  return nanstd(X, **kwArgs) * nansdec(X, **kwArgs)

#-----------------------------------------------------------------------------------------------------------------------
def nanvrvr(X, **kwArgs):
  n = nandem(X, **kwArgs)
  v = nanvar(X, **kwArgs)
  k = nankurt(X, **kwArgs)
  v2 = v ** 2.0
  kv2 = k * v2
  return (n-1.)/(n**2.0-2.*n+3.) * (kv2 - v2*(n-3.)/(n-1))

#-----------------------------------------------------------------------------------------------------------------------
def nanfloat(X, _reg = None):
  d = nDim(X)
  reg = True if _reg is None else _reg
  if d == 0:
    Y = np.nan
    if isnum(X):
      Y = float(X)
  elif d == 1:
    n = len(X)
    Y = np.empty(n, dtype = float)
    for i in range(n):
      Y[i] = nanfloat(X[i])
    reg = True
  else:
    n = len(X)
    m = np.empty(n, dtype = int)
    Y = [[]] * n
    for i in range(n):
      Y[i], reg = nanfloat(X[i], reg)
      try:
        m[i] = len(Y[i])
      except TypeError: # in case a dimension is `skipped'
        reg = False
    reg = reg and m.min() == m.max()
    if reg: Y = np.array(Y)
  if _reg is None:
    return Y
  return Y, reg

#-----------------------------------------------------------------------------------------------------------------------
def nancopy(X):
  isarr = isarray(X)
  if not(isarr):
    if isNaN(X):
      return np.NaN
    else:
      return X
  istup = type(X) is tuple
  Y = list(X) if istup else X[:]
  n = len(Y)
  for i in range(n):
    Y[i] = nancopy(Y[i])
  if istup: return tuple(Y)
  return Y

#-----------------------------------------------------------------------------------------------------------------------
def multi_axis_func(X, _a = [], axfunc = nancnt):
  if Type(_a) is int:
    a = list(range(np.ndim(X)))
    del a[_a]
    return multi_axis_func(X, a, axfunc)
  elif not(len(_a)):
    return axfunc(X.ravel())
  a = np.unique(_a)[::-1]
  s = list(X.shape)
  x = X.copy()
  for a_ in a:
    x = axfunc(x, axis = a_)
    s[a_] = 1
  return x.reshape(s)

#-----------------------------------------------------------------------------------------------------------------------
def Max(X, _a = []):
  return multi_axis_func(X, _a, np.max)

#-----------------------------------------------------------------------------------------------------------------------
def Min(X, _a = []):
  return multi_axis_func(X, _a, np.min)

#-----------------------------------------------------------------------------------------------------------------------
def Sum(X, _a = []):
  return multi_axis_func(X, _a, np.sum)

#-----------------------------------------------------------------------------------------------------------------------
def Mean(X, _a = []):
  return multi_axis_func(X, _a, np.mean)

#-----------------------------------------------------------------------------------------------------------------------
def reptile(x, X, a = 0):
  n = len(x.ravel())
  s = list(X) if type(X) is tuple else X.shape
  if s[a] != n: raise ValueError("Input dimensions incommensurate.")
  N = len(s)
  res = np.ones(N, dtype = int)
  rep = np.array(s, dtype = int)
  res[a] = n
  rep[a] = 1
  return np.tile(x.reshape(res), rep)

#-----------------------------------------------------------------------------------------------------------------------
def ents(X, base = None):
  x = unzero(np.fabs(X))
  if base is None:
    return -X*np.log(x)
  elif base == 2:
    return -X*np.log2(x)
  elif base == 10:
    return -X*np.log10(x)
  return -X*np.log(x, base)

#-----------------------------------------------------------------------------------------------------------------------
def entropy(X, base = None, **kwargs):
  return np.sum(ents(X, base), **kwargs)

#-----------------------------------------------------------------------------------------------------------------------
def mutual(X, base = None):
  je = np.sum(np.ravel(ents(X, base)))
  nd = np.ndim(X)
  me = 0.
  for i in range(nd):
    me += entropy(Sum(X, i), base)
  return  me - je

#-----------------------------------------------------------------------------------------------------------------------
def geomean(_X, **kwargs):
  mino = 1e-300
  m = np.nonzero(_X < 0.0); m = m[0]
  isneg = len(m) > 0
  if isneg and len(m) != len(_X.ravel()):
    raise ValueError("geomean: Heterogeneous sign amongst elements of input array.")
  X = -_X if isneg else _X
  m = np.exp(np.mean(np.log(X + mino), **kwargs))
  if isneg: m = -m
  return m

#-----------------------------------------------------------------------------------------------------------------------
def geostd(_X, **kwargs):
  mino = 1e-300
  m = np.nonzero(_X < 0.0); m = m[0]
  isneg = len(m) > 0
  if isneg and len(m) != len(_X.ravel()):
    raise ValueError("geomean: Heterogeneous sign amongst elements of input array.")
  X = -_X if isneg else _X
  s = np.exp(np.std(np.log(X + mino), **kwargs))
  return s

#-----------------------------------------------------------------------------------------------------------------------
def meanabsres(hy, y, **kwargs):
  return np.mean(np.fabs(hy - y), **kwargs)

#-----------------------------------------------------------------------------------------------------------------------
def sumsqrdif(x, X):
  ssd = np.zeros(X.shape, dtype = float)
  nx = len(x)
  for i in range(len(x)):
    ssd += (x[i] - X) ** 2.0

  return ssd

#-----------------------------------------------------------------------------------------------------------------------
def stderr(x, **kwargs):
  mino = 1e-300
  y = np.std(x, **kwargs)
  ny = len(y.ravel())
  n = len(x.ravel()) / ny if ny else 0
  return y / (np.sqrt(float(n))+mino)

#-----------------------------------------------------------------------------------------------------------------------
def fractile(a, q, **kwargs):
  if type(a) is list: a = np.array(a, dtype = float)
  if type(q) is list or type(q) is tuple: q = np.array(q, dtype = float)
  q *= 100
  if type(q) is np.ndarray:
    return np.percentile(a, list(q), **kwargs)
  else:
    return np.percentile(a, q, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------
def iqr(x, **kwargs):
  if x.size:
    return fractile(x, 0.75, **kwargs) - fractile(x, 0.25, **kwargs)
  else:
    return 0.

#-----------------------------------------------------------------------------------------------------------------------
def ranfrac(x, f = 0.5):
  if type(f) is list or type(f) is tuple: f = np.array(f, dtype = float)
  minx = x.min()
  maxx = x.max()
  ranx = maxx - minx
  return minx + f*ranx

#-----------------------------------------------------------------------------------------------------------------------
def indfrac(y, _f = 0.5, pol = None, opts = False): # if opts = True, returns both
  if isnum(_f): _f = np.array([_f], dtype = float)
  n, N = len(y), len(_f)
  x = np.arange(n, dtype = float)
  if pol is None:
    cc = np.corrcoef(x, y)[0,1]
    pol = -1. if cc < 0. else +1.
  if pol > 0:
    More = np.greater
    f = np.array(_f)
  else:
    More = np.less
    f = 1. - np.array(_f)
  F = ranfrac(y, f)
  I = np.zeros(N, dtype = int)
  for i in range(N):
    b = np.nonzero(More(y, F[i]))[0]
    if len(b):
      I[i] = b[0]
    elif f[i] > 0.5:
      I[i] = n
  if opts: return I, F
  return I
#-----------------------------------------------------------------------------------------------------------------------
def findfrac(_y, _f = 0.5, spec = None, opts = False, **kwds):
  """Returns indices of fractile crossings (default _f = 0.5) in sequence _y
  spec = None or [0,0]: automates crossing direction and uses entire range
  spec = [0, 1]: detects increasing crossings using entire range
  spec = [0, -1]: detects decreasing crossings using entire range
  spec = [-1, 0]: automated direction but taking range from averaged value preceding min/max
  spec = [+1, 0]: automated direction but taking range from averaged value succceding min/max
  opts = False: exports indices only.
  opts = True: exports indices, fractiles

  """
  f = np.array([_f], dtype = float) if isnum(_f) else _f
  y = np.array(_y, dtype = float) if type(_y) is list else _y
  if isnum(_f): _f = np.array(_f, dtype = float)
  if spec is None: spec = 0
  if isnum(spec) is int: spec = (spec, 0)
  spec = list(np.array(spec, dtype = int))
  n, N = len(y), len(f)
  x = np.arange(n, dtype = float)
  if spec[1] == 0:
    cc = np.corrcoef(x, y)[0,1]
    spec[1] = -1 if cc < 0. else +1
  if spec[1] > 0:
    Amax = np.argmax
    Amin = np.argmin
  else:
    Amax = np.argmin
    Amin = np.argmax
  M = np.ones(n, dtype = bool)
  m = np.array( [y.min(), y.max()], dtype = float)
  if spec[0] != 0:
    if spec[0] < 0:
      i = Amin(y)
      M[:i] = False
    else:
      i = Amax(y)
      j = i + 1
      if j < n: M[j:] = False
    i = 1 if np.sign(spec[0]) == np.sign(spec[1]) else 0
    W = np.logical_not(M)
    if np.any(W): m[i] = np.median(y[W])
  F = ranfrac(m, f)
  I = np.ones(N, dtype = float)
  for i in range(N):
    Fi = F[i]
    if spec[1] > 0:
      b = y > Fi
    else:
      b = y < Fi
    b = np.logical_and(b, M)
    b = np.nonzero(b)
    k = len(b)-1 if spec[0] > 0 else 0
    b = b[k]
    if len(b):
      b = b[0]
      if b == 0:
        I[i] = float(b)
      else:
        d = b - 1
        lo, hi = y[d], y[b]
        d, b = float(d), float(b)
        I[i] = (d*np.fabs(hi - F[i]) + b*np.fabs(lo - F[i])) / unzero(np.fabs(hi - lo))
    elif f[i] > 0.5:
      I[i] = float(n-1)
  if opts: return I, F
  return I

#-----------------------------------------------------------------------------------------------------------------------
def numspace(start = 0.0, finish = 1.0, n = 50, spec = 0):
  if isint(spec):
    if spec == 0:
      return arispace(start, finish, n)
    elif spec == -1:
      return oddspace(start, finish, n)
    elif spec == -2:
      return arcspace(start, finish, n)
    elif spec == 1:
      return geospace(start, finish, n)
    elif spec == 2:
      return powspace(start, finish, n, 0.5)
    else:
      raise ValueError("numspace: Unknown space specification.")
  else:
    return powspace(start, finish, n, float(spec))

#-----------------------------------------------------------------------------------------------------------------------
def arispace(_start = 0.0, _finish = 1.0, n = 50):  # arithmetic space, but allowing interval specification more usefully than arange for floating point
  if isint(n): return np.linspace(_start, _finish, n)
  start, finish = min(_start, _finish), max(_start, _finish)
  x = start + n*np.arange(1+np.ceil( (finish-start)/ n), dtype = float)
  if n > 1:
    if (x[-2] <= n) and (x[-1] > n):
      x = x[:-1]
    elif (x[-2] >= n) and (x[-1] < n):
      x = x[:-1]
  if _start > _finish:
    return x[::-1]
  else:
    return x

#-----------------------------------------------------------------------------------------------------------------------
def geospace(start = 1.0, finish = np.e, n = 50): # geometric space, but allowing interval specification more usefully than arange for floating point
  mino = 1e-300
  pol = np.sign(start)
  if pol != np.sign(finish):
    raise ValueError("Start and finish input arguments must have identical polarity")
  lims =  np.array((abs(start), abs(finish)), dtype = float)
  lims.sort()
  if isfloat(n): n = np.log(abs(n))
  y = pol * np.exp(arispace(np.log(abs(lims[0]+mino)), np.log(abs(lims[1])+mino), n))
  if abs(start) > abs(finish):
    return y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def powspace(start = 0.0, finish = 1.0, n = 50, pwr = 0.5): # square reciprocal space (for prior x^{-2})
  pol = np.sign(start)
  if pol != np.sign(finish):
    raise ValueError("Start and finish input arguments must have identical polarity")
  p = float(pwr)
  q = 1.0 / p
  lims =  np.array((start**p, finish**p), dtype = float)
  lims.sort()
  y = np.linspace(lims[0], lims[1], n) ** q
  if abs(start) > abs(finish):
    return y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def oddspace(start = 0.0, finish = 1.0, n = 50): # geometric odds (Haldane) space
  mino = 1e-300
  pol = np.sign(start) + np.sign(finish)
  if abs(pol) > 1.5:
    pol /= 2.0
  elif abs(pol) < 0.5:
    raise ValueError("Start and finish input arguments must have identical \polarity")
  lims =  np.array((abs(start), abs(finish)), dtype = float)
  lims.sort()
  if lims.max() > 1.0:
    raise ValueError("Absolute values of start and finish input arguments must not exceed unity.")
  lolims = np.log(lims/((1.0 - lims) + mino)+mino)
  o = np.exp(np.linspace(lolims[0], lolims[1], n))
  y = o / (o+1.0)
  if abs(start) > abs(finish):
    return y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def arcspace(start = 0.0, finish = 1.0, n = 50): # arsine pdf (=arcsin root space)
  pol = np.sign(start) + np.sign(finish)
  if abs(pol) > 1.5:
    pol /= 2.0
  elif abs(pol) < 0.5:
    raise ValueError("Start and finish input arguments must have identical polarity")
  lims =  np.array((abs(start), abs(finish)), dtype = float)
  lims.sort()
  if lims.max() > 1.0:
    raise ValueError("Absolute values of start and finish input arguments must not exceed unity.")
  #cdflims = stats.arcsine.cdf(lims)
  #cdf = np.linspace(cdflims[0], cdflims[1], n)
  #y = pol * stats.arcsine.ppf(cdf)
  thlims = np.arcsin(np.sqrt(lims))
  th = np.linspace(thlims[0], thlims[1], n)
  y = np.sin(th) ** 2.0
  if abs(start) > abs(finish):
    return y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def expnspace(start = 0.0, finish = 1.0, n = 50, _m = 1.): # exponential negative (i.e. decay) space at mean m
  m = np.fabs(_m)
  pol = np.sign(start) + np.sign(finish)
  if abs(pol) > 1.5:
    pol /= 2.0
  elif abs(pol) < 0.5:
    raise ValueError("Start and finish input arguments must have identical polarity")
  lims =  np.array((abs(start), abs(finish)), dtype = float)
  lims.sort()
  exlims = -np.exp(-lims/m)
  ex = np.linspace(exlims[0], exlims[1], n)
  y = pol*np.fabs(-m*np.log(-ex))
  if abs(start) > abs(finish):
    return y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def tanhspace(start = -1., finish = 1., n = 50, mid = 0.):
  Wid = np.pi
  if n < 3: return np.linspace(start, finish, n)
  wid = 0.5 * (finish - start)
  lohi = np.array([start, finish], dtype = float)
  lohi = np.tanh(Wid) * (lohi - mid) / unzero(wid)
  t = np.linspace(np.tanh(lohi[0]), np.tanh(lohi[1]), n+2)[1:-1]
  x = np.arctanh(t)
  minx, maxx = np.min(x), np.max(x)
  return start + (finish-start) * (x - minx) / unzero(maxx - minx)
#-----------------------------------------------------------------------------------------------------------------------
def normspace(start = -1., finish = 1., n = 50, mn = 0., sd = None):
  if n < 3: return np.linspace(start, finish, n)
  if sd is None: sd = np.fabs(finish-start) / 6.
  lo = stats.norm.cdf(start, loc = mn, scale = sd)
  hi = stats.norm.cdf(finish, loc = mn, scale = sd)
  t = np.linspace(lo, hi, n)
  x = stats.norm.ppf(t, loc=mn, scale=sd)
  x[0], x[-1] = start, finish
  return x

#-----------------------------------------------------------------------------------------------------------------------
def distribute(_x, _lo = None, _hi = None, n = 'all', func = 0, *args): # with `replacement'
  x = _x if type(_x) is np.ndarray else np.array(_x)
  lo = float(_lo) if _lo is not None else x.min()
  hi = float(_hi) if _hi is not None else x.max()
  X = None
  if n == 'all':
    xok = np.nonzero(np.logical_and(x >= lo, x <= hi))[0]
    n = len(xok)
    X = np.sort(x[xok])
  elif type(func) is not str:
    _func = {-2:arcspace, -1:oddspace, 0:arispace, 1:geospace, 2:powspace}
    func = _func[func]
  if X is None: X = func(lo, hi, n, *args)
  I = np.empty(n, dtype = int)
  i = 0
  done = n <= 0
  while not(done):
    d = np.fabs(X[i] - x)
    k = d.argmin()
    K = np.nonzero(d == d[k])[0]
    for j in range(len(K)):
      I[i] = K[j]
      i += 1
    done = i == n
  i = np.argsort(x[I])
  return I[i]

#-----------------------------------------------------------------------------------------------------------------------
def opthistbw(_x):
  '''  Uses optimisation method of Shimazaki and Shinomoto with sensible bounds :
  [minbw = range/length, maxbw = range/sqrt(length))]
  @article{shimazaki2007method,
    title={A method for selecting the bin size of a time histogram},
    author={Shimazaki, H. and Shinomoto, S.},
    journal={Neural Computation},
    volume={19},
    number={6},
    pages={1503--1527},
    year={2007},
    publisher={MIT Press}
  }
  '''
  MD = False
  if type(_x) is list:
    MD = True
  elif type(_x) is np.ndarray:
    if _x.ndim > 1:
      MD = True
  if MD:
    n_x = len(_x)
    bw = np.empty(n_x, dtype = float)
    for k in range(n_x):
      bw[k] = opthistbw(_x[k])
    return np.mean(bw)
  x = nanravel(_x);
  n = len(x);
  minx = x.min();
  maxx = x.max();
  ranx = maxx - minx;
  if n < 3 or ranx < 1e-300: return ranx
  minb = ranx / float(n)
  maxb = ranx / np.sqrt(float(n));
  b = geospace(minb, maxb, n);
  m = np.array(np.ceil(b/minb), dtype = int) + int(np.sqrt(float(n)));
  C = np.empty((n), dtype = float)
  for i in range(n):
    mi = m[i]
    m2 = mi / 2
    bi = b[i];
    b2 = 0.5 * bi
    bs = bi ** 2.0
    nb = int(round((ranx+bi)/bi))
    b0 = np.linspace(minx-b2, maxx+b2, nb)
    db = np.linspace(-b2, b2, mi)
    c = np.zeros(mi, dtype = float)
    for j in range(mi):
      h = np.histogram(x, b0 + db[j])[0]
      lh  = float(len(h))
      if j < m2:
        h = h[:-1]
        lh -= 1.
      elif j > m2:
        h = h[1:]
        lh -= 1.
      mh = np.mean(h)
      vh = np.sum( (h-mh)**2.0 )/lh
      c[j] = (2.*mh - vh) / bs
    C[i] = np.mean(c)
  bw = b[np.argmin(C)]
  return bw;

#-----------------------------------------------------------------------------------------------------------------------
def augment(_x, sig = 1e-15, eps = 1e-300):
  n = len(_x)
  if n<=1: return _x
  x = np.copy(_x)
  i, j = 0, 0
  if x[0] > x[n-1]:
    i = n-1
  else:
    j = n-1
  if x[i] > 0.:
    x[i] *= (1.-sig)
  elif x[i] < 0.:
    x[i] *= (1.+sig)
  else:
    x[i] = -eps
  if x[j] > 0.:
    x[j] *= (1.+sig)
  elif x[j] < 0.:
    x[j] *= (1.-sig)
  else:
    x[j] = eps
  return x

#-----------------------------------------------------------------------------------------------------------------------
def binspace(lo = 0., hi = 1., d = 50, spec = 0):
  if isint(d): return numspace(lo, hi, d, spec)
  hilo = lo > hi
  if not(spec):
    if hilo:
      x = hi + d*np.arange(1+np.ceil((lo-hi)/d), dtype = float)[::-1]
      n = len(x)
      x = augment(x)
      if n < 2: return x
      if x[1] > lo: return x[1:]
      return x
    else:
      x = lo + d*np.arange(1+np.ceil((hi-lo)/d), dtype = float)
      n = len(x)
      if n < 2: return x
      if x[n-2] > hi: return x[:(n-1)]
      return x
  slo = np.sign(lo)
  shi = np.sign(hi)
  if slo != shi:
    raise ValueError("Log specification requires matched signs for limits")
  if d == 1.:
    raise ValueError("Unity separation with log specification divides by zero.")
  y = augment(np.exp(binspace(np.log(np.fabs(lo)), np.log(np.fabs(hi)), np.log(np.fabs(d)))))
  if slo < 0.: return -y
  return y

#-----------------------------------------------------------------------------------------------------------------------
def cumspace(_n):
  n = float(_n)
  d = 0.48/n - 0.2/(n**1.3) - 0.1/(n**6.)
  return np.linspace(d, 1.-d, int(n))

#-----------------------------------------------------------------------------------------------------------------------
def histbins(x, bw = None):
  if bw is None: bw = opthistbw(x)
  return arispace(np.min(x), np.max(x) + bw, bw)

#-----------------------------------------------------------------------------------------------------------------------
def naninterp(_x, left = None, right = None, inplace = False, acceptreturnednans = False):
  x = _x if inplace else _x.copy()
  if not(np.ndim(x)):
    if isNaN(x):
      if acceptreturnednans:
        return x
      else:
        raise ValueError("Cannot interpolate a NaN scalar")
    else:
      return x;
  i = np.arange(len(x), dtype = float)
  nanx = isNaN(x)
  j = np.nonzero(nanx)[0]
  k = np.nonzero(~nanx)[0]
  if not(len(k)):
    if acceptreturnednans:
      return x
    else:
      raise ValueError("Cannot interpolate an array solely comprising of NaN values")
  if not(len(j)):
    return x
  if left is None: left = np.nanmin(x)
  if right is None: right = np.nanmax(x)
  x[j] = np.interp(i[j], i[k], x[k])
  return x

#-----------------------------------------------------------------------------------------------------------------------
def naninterp2(X, left = None, right = None, inplace = False, recursive = True):
  Y = X if inplace else X.copy()
  nY = isNaN(Y)
  nYsum = nY.sum()
  if nYsum < 1: return Y
  sY = Y.shape
  rY = sY[0]
  cY = 1 if len(sY) < 2 else sY[1]
  NY = np.array(np.logical_not(nY), dtype = float)
  NY[NY < 1] = np.NaN
  N0 = NY * np.tile(np.arange(rY).reshape(rY, 1), (1, cY)) # Valid indices of each column
  N1 = NY * np.tile(np.arange(cY).reshape(1, cY), (rY, 1)) # Valid indices of each row
  N0min, N0max = np.nanmin(N0, axis = 0), np.nanmax(N0, axis = 0) # Valid extremes for each column
  N1min, N1max = np.nanmin(N1, axis = 1), np.nanmax(N1, axis = 1) # Valid extremes for each row
  iY = np.nonzero(nY)
  nY0 = nY.sum(axis = 0) # count of NaNs in each column
  nY1 = nY.sum(axis = 1) # count of NaNs in each row
  NY0 = rY - nY0 # count of valid data in each column
  NY1 = cY - nY1 # count of valid data in each row
  Y0 = np.empty(sY, dtype = float) # interpolating each row
  for i in range(rY):
    Y0[i, :] = naninterp(Y[i, :], left, right, False, True)
  Y1 = np.empty(sY, dtype = float) # interpolating each column
  if cY > 1:
    for j in range(cY):
      Y1[:, j] = naninterp(Y[:, j], left, right, False, True)
  minW = min(rY, cY)
  minw = 3
  while minW > 0 and nYsum == isNaN(Y).sum():
    minW -= 1
    minw = min(minw, max(1, minW-1))
    for k in range(len(iY[0])):
      i, j = iY[0][k], iY[1][k]
      y0, y1 = Y0[i][j], Y1[i][j]
      y0nan, y1nan = isNaN(y0), isNaN(y1)
      w0 = NY1[i] # relative weight of interpolation within respective row
      w1 = NY0[j] # relative weight of interpolation within respective column
      n0min, n0max = N0min[j], N0max[j] # Valid extremes for respective column
      n1min, n1max = N1min[i], N1max[i] # Valid extremes for repective row
      if y0nan and not(y1nan): # interpolated value only within the row
        if (j == 0 or j == cY-1) or (w1 > minw and (j >= n1min and j <= n1max)):
          Y[i][j] = y1
      elif not(y0nan) and y1nan: # interpolated value only within the column
        if (i == 0 or i == rY-1) or (w0 > minw and (i >= n0min and i <= n0max)):
          Y[i][j] = y0
      elif not(y0nan) and not(y1nan):
        if (i == 0 or i == rY-1) or (w0 > 1 and w1 < 2):
          Y[i][j] = y0
        elif (j == 0 or j == cY-1) or (w0 < 2 and w1 > 1):
          Y[i][j] = y1
        else:
          if (j < n1min or j > n1max): w0 = 0
          if (i < n0min or i > n0max): w1 = 0
          if w0 + w1 > minW:
            Y[i][j] = (float(w0)*y0 + float(w1)*y1)/float(w0+w1)
  if recursive:
    if (nYsum == isNaN(Y).sum()):
      return Y
    else:
      return naninterp2(Y, left, right, True, True)
  return Y

#-----------------------------------------------------------------------------------------------------------------------
def extarray(x, n = 1, logscale = False):
  nx = len(x)
  minx, maxx = x.min(), x.max();
  sign = 0;
  if (minx <= 0 and minx >= 0):
    if logscale:
      raise ValueError('extarray: Cannot apply log-scale to mixed sign data.')
  if logscale:
    if minx < 0:
      sign = -1
      minx, maxx = log(-minx), log(-maxx)
    else:
      sign = 1
      minx, maxx = log(minx), log(maxx)
  midx = 0.5 * (maxx + minx)
  if n == 1 or nx == 1:
    y = np.tile(midx, max(n, nx))
  else:
    widx = 0.5 * (maxx - minx)
    widy = widx * float(n-1) / unzero(float(nx-1))
    y = np.linspace(midx-widy, midx+widy, n)
  if logscale:
    y = exp(y) if sign > 0 else -exp(y)
  if (x[0] > x[-1] and y[0] < y[-1]) or (x[0] < x[-1] and y[0] > y[-1]):
    y = y[::-1]
  return y

#-----------------------------------------------------------------------------------------------------------------------
def midlims(n, d = 1, minm = -np.inf, maxm = np.inf):
  i = max(minm, int(0.5 * float(n-d)))
  j = min(maxm, i + d + 1)
  return i, j

#-----------------------------------------------------------------------------------------------------------------------
def midarray(x, n = 1):
  nx = len(x)
  if (n >= nx): return extarray(x, n)
  i, j = midlims(nx, n, 0, nx)
  return x[i:j]

#-----------------------------------------------------------------------------------------------------------------------
def midarray2(x, nm = [1, 1]):
  n, m = nm
  X = midarray(x, n)
  nX = len(X)
  for i in range(nX):
    if not(i):
      y = midarray(X[i], m)
      ny = len(y)
      Y = np.empty( (nX, ny), dtype = float)
      Y[i] = y
    else:
      Y[i] = midarray(X[i], m)

#-----------------------------------------------------------------------------------------------------------------------
def wrapsumarray(x, d):
  # For 2D inputs, specification d[i] = 0 prevents wrapping in the ith dimension
  if isint(d):
    if d == 0: return x
    n = len(x)
    i, j = midlims(n, d, 0, n)
    y = np.copy(x[i:j])
    l = x[:i]
    h = x[j:]
    if len(y) == 1:
      y += np.sum(h) + np.sum(l)
    else:
      if len(h): y[:len(h)] += h
      if len(l): y[-len(l):] += l
    return y
  if len(d) != 2:
    raise ValueError("Presently only able to deal with two dimensions.")
  X = np.copy(x)
  if d[0]:
    XT = X.T
    n = len(XT)
    _X = [[]] * n
    for i in range(n):
      _X[i] = wrapsumarray(XT[i], d[0])
    X = np.array(_X, dtype = float).T
  if d[1]:
    n = len(X)
    _X = [[]] * n
    for i in range(n):
      _X[i] = wrapsumarray(X[i], d[1])
    X = np.array(_X, dtype = float)
  return X

#-----------------------------------------------------------------------------------------------------------------------
def padarray(x, n = 0, v = 0., padfront = False):
  y = x.copy()
  if not(np.ndim(y.shape)):
    nx = 1
  else:
    nx = len(y)
  if nx >= n: return y
  z = np.array( (v), dtype = y.dtype)
  if padfront:
    z = np.concatenate( (np.tile(z, n - nx), y) )
  else:
    z = np.concatenate( (y, np.tile(z, n - nx)) )
  return z

#-----------------------------------------------------------------------------------------------------------------------
def expo(x, maxx = 709.): # handles overflows
  if Type(x) is float:
    if x < maxx: return np.exp(x)
    return np.inf
  o = x < maxx
  if np.all(o): return np.exp(x)
  y = np.tile(np.inf, x.shape)
  y[o] = np.exp(x[o])
  return y

#-----------------------------------------------------------------------------------------------------------------------
def sqeucldis(_X, _Y = None, nd = None): # output dimensions: (X.shape[0], Y.shape[0])
  if _Y is None: _Y = _X
  X, Y = nparray2D(_X, dtype = float), nparray2D(_Y, dtype = float)
  rX, cX = X.shape
  rY, cY = Y.shape

  if nd is not None:
    if len(nd) != cX:
      raise ValueError("Normalisation denomator argument length must match columns of input data")
  if cX != cY:
    raise ValueError("Column dimensions incommensurate.")

  X3 = np.tile(X.T.reshape((cX, rX, 1)), (1, 1, rY))
  Y3 = np.tile(Y.T.reshape((cY, 1, rY)), (1, rX, 1))
  sq =  (X3 - Y3)**2.
  if nd is not None:
    sq = np.sqrt(sq)
    nd = np.tile(nd.reshape((cX, 1, 1), (1, rX, rY)))
    sq /= unzero(nd)
    sq = sq**2.
  ss = sq.sum(axis = 0)
  sed = ss.reshape((rX, rY))
  return sed

#-----------------------------------------------------------------------------------------------------------------------
def isconvex(X, o = 2):
  nd = X.ndim
  More = np.less if o % 2 else np.greater
  if nd < 1:
    return []
  n = len(X)
  if nd == 1:
    if n <= o: return False
    return More(np.polyfit(np.arange(n, dtype = float), X, o)[0], 0.)
  N = len(X[0])
  if N <= o: return np.tile(False, n)
  x = np.arange(N, dtype = float)
  b = np.empty(n, dtype = bool)
  for i in range(n):
    b[i] = More(np.polyfit(x, X[i], o)[0], 0.)
  return b

#-----------------------------------------------------------------------------------------------------------------------
def isconcave(X, o = 2):
  nd = X.ndim
  Less = np.greater_equal if o % 2 else np.less_equal
  if nd < 1:
    return []
  n = len(X)
  if nd == 1:
    if n <= o: return True
    return Less(np.polyfit(np.arange(n, dtype = float), X, o)[0], 0.)
  N = len(X[0])
  if N <= o: return np.tile(False, n)
  x = np.arange(N, dtype = float)
  b = np.empty(n, dtype = bool)
  for i in range(n):
    b[i] = Less(np.polyfit(x, X[i], o)[0], 0.)
  return b

#-----------------------------------------------------------------------------------------------------------------------
def padarray2(x, nm = [0, 0], v = 0., padtop = False, padleft = False):
  n, m = nm
  nx = len(x)
  if not(nx): return y
  mx = len(x[0])
  dr, dc = max(0, n - nx), max(0, m - mx)
  if (not(dr) and not(dc)): return np.copy(x)
  n, m = max(n, nx), max(m, mx)
  y = np.tile(v, (n, m))
  j = dr if dr > 0 and padtop else 0
  for i in range(nx):
    k = i + j
    y[k] = padarray(x[i], m, v, padleft)
  return y


#-----------------------------------------------------------------------------------------------------------------------
def middle(x, **kwds):
  minx = x.min(**kwds)
  maxx = x.max(**kwds)
  return 0.5 * (minx + maxx)

#-----------------------------------------------------------------------------------------------------------------------
def gckmode(_x, _X = None, _resn = None, _stdc = None):
  x = nparray(_x, dtype = float)
  X = x if _X is None else _X
  if _resn is None: _resn = max(len(x), 117649)
  if len(x) == 1: return x[0]
  gck = gckernel()
  gck.setResn(_resn)
  if _stdc is not None: gck.setKernel(None, _stdc)
  gck.convolve(X)
  return gck.x[np.argmax(gck.p)]

#-----------------------------------------------------------------------------------------------------------------------
def gcklhmf(_x, _X = None, _resn = None, _stdc = None):
  x = nparray(_x, dtype = float)
  X = x if _X is None else _X
  if _resn is None: _resn = max(len(X), 117649)
  gck = gckernel()
  gck.setResn(_resn)
  if _stdc is not None: gck.setKernel(None, _stdc)
  gck.convolve(X)
  i = arghist(x, gck.x)
  p = gck.p[i]
  figure()
  plot(gck.x, gck.p)
  return p / unzero(p.sum())

#-----------------------------------------------------------------------------------------------------------------------
def centrescale(x):
  meanx = np.mean(x);
  stdvx = unzero(np.std(x))
  return (x - meanx) / stdvx
#-----------------------------------------------------------------------------------------------------------------------
def middlerange(x):
  minx, maxx = x.min(), x.max()
  midx, widx = 0.5 * (maxx + minx), unzero(0.5 * (maxx - minx))
  return (x-midx) / widx

#-----------------------------------------------------------------------------------------------------------------------
class gckernel:
  defresn1 = 117649
  defresn2 = 343
  ND = 0
  N = 0
  M = 0
  defstdw = [3., 3.]
  defstdc = [0, 0]
  # if stdw[i] == np.inf, the ith dimension is treated as polar
  def __init__(self, _logsca = [False, False], _resn = [0, 0], _resd = [0.,0.], _stdw = None, _stdc = None):
    self.initialise(_logsca, _resn, _resd, _stdw, _stdc)
  def initialise(self, _logsca = [False, False], _resn = [0, 0], _resd = [0.,0.], _stdw = None, _stdc = None):
    self.setResn(_resn)
    self.setResd(_resd)
    self.setKernel(_stdw, _stdc)
    self.setLogsca(_logsca)
    self.setData()
    self.setBins()
    self.genKernel()
  def setResn(self, _resn = [0, 0]):
    if not(type(_resn) is list): _resn = [_resn]
    self.resn = np.array(_resn, dtype = int)
  def setResd(self, _resd = [0., 0.]):
    if not(type(_resd) is list): _resd = [_resd]
    self.resd = np.array(_resd, dtype = float)
  def setKernel(self, _stdw = None, _stdc = None):
    if _stdw is None: _stdw = self.defstdw
    if _stdc is None: _stdc = self.defstdc
    self.stdw = nparray(_stdw, dtype = float)
    self.pola = np.isinf(self.stdw)
    self.ncx, self.ncy = 0, 0
    self.stdc = nparray(_stdc, dtype = float)
  def genbins(self, dimn, defres, minv, maxv):
    if (self.resd[dimn]>0.):
      b = numspace(minv, maxv, self.resd[dimn], int(self.logsca[dimn]))
    else:
      if self.resn[dimn] > 1:
        if self.xx is None:
          b = numspace(minv, maxv, self.resn[dimn], int(self.logsca[dimn]));
        else:
          b = self.xx
      else:
        b = numspace(minv, maxv, defres, int(self.logsca[dimn]));
      self.resd[dimn] = 0.;
    self.resn[dimn] = len(b)
    if self.resn[dimn] < 2: return b
    if (self.logsca[dimn]):
      self.resd[dimn] = exp(log(fabs(b[1])) - log(fabs(b[0])))
    else:
      self.resd[dimn] = b[1] - b[0]
    return b
  def calcBinw(self, dimn, b):
    bw = 0.
    nb = len(b)
    if (nb < 2): return bw
    if (self.logsca[dimn]):
      bw = exp(log(b[1]) - log(b[0]))
    else:
      bw = b[1] - b[0]
    return bw
  def genBins(self):
    if not(self.ND) and self.xx is None: return
    if not(self.xx is None):
      self.resn[0] = len(self.xx)
      self.resd[0] = self.calcBinw(0, self.xx)
    if not(self.yy is None):
      self.resn[1] = len(self.yy)
      self.resd[1] = self.calcBinw(1, self.yy)
    if (self.ND < 1): return
    defres = 0
    if (self.ND == 1): defres = self.defresn1;
    if (self.ND == 2): defres = self.defresn2;
    if (self.xx is None):
      if self.pola[0]:
        self.xx = self.genbins(0, defres, -np.pi, np.pi)
      else:
        self.xx = self.genbins(0, defres, self.minx, self.maxx)
    if (self.ND < 2): return
    if (self.yy is None):
      if self.pola[1]:
        self.yy = self.genbins(1, defres, -np.pi, np.pi)
      else:
        self.yy = self.genbins(1, defres, self.miny, self.maxy)
  def calcStds(self):
    if self.ND < 1: return
    # NB/ VM concentration estimates are inversily proportional to the variance in general
    if (self.M > 1):
      self.stdx = np.empty(self.M, dtype = float)
      for i in range(self.M):
        if self.logsca[0]:
          self.stdx[i] = nanstd(np.log(self.XX[i]))
        elif self.pola[0]:
          self.stdx[i] = ccon(self.XX[i])
        else:
          self.stdx[i] = nanstd(self.XX[i])
      return
    if (self.logsca[0]):
      self.stdx = nanstd(np.log(self.X))
    elif self.pola[0]:
      self.stdx = ccon(self.X)
    else:
      self.stdx = nanstd(self.X)
    if self.ND < 2: return
    if (self.logsca[1]):
      self.stdy = nanstd(np.log(self.Y))
    elif self.pola[1]:
      self.stdy = ccon(self.Y)
    else:
      self.stdy = nanstd(self.Y)
  def setLogsca(self, _logsca = [False, False]):
    self.logsca = _logsca[:]
    if ((self.logsca[0] and self.pola[0]) or (self.logsca[1] and self.pola[1])):
      print("Warning: polar axis specified on logarithmic scale: this will fail.")
    if (self.ND > 0):
      self.genBins()
      self.calcStds()
  def useKernel(self, _C = None):
    self.C = _C;
  def genKernel(self, _ncx = 0, _ncy = 0):
    if not(self.ND): return
    if self.M > 0:
      if (len(self.stdc) < self.M): self.stdc = np.tile(self.stdc[0], self.M)
      if (np.array(self.stdc).min() <= 0.):
        self.calcStds()
        for i in range(self.M):
          if (self.stdc[i] <= 0.):
            if self.pola[0]:
              self.stdc[i] = self.stdx[i] * (float(len(self.XX[i]))**(.6)) # a la Bhumbra and Dyball 2010
            else:
              self.stdc[i] = self.stdx[i] * (float(len(self.XX[i]))**(-.3)) # a la Bhumbra and Dyball 2010
      if not(_ncx):
        if self.logsca[0]:
          self.ncx = np.ceil(2. * self.stdw[0] * self.stdc.max() / np.log(self.resd[0]))
        elif self.pola[0]:
          self.ncx = np.ceil(0.5*self.resn[0])
        else:
          self.ncx = np.ceil(2. * self.stdw[0] * self.stdc.max() / self.resd[0])
      else:
          self.ncx = _ncx
      self.ncx = int(self.ncx)
      self.C = np.empty( (self.M, self.ncx), dtype = float)
      if self.pola[0]:
        self.cx = np.linspace(-0.5*np.pi, 0.5*np.pi, self.ncx)
      else:
        self.cx = np.linspace(-0.5*float(self.ncx), 0.5*float(self.ncx), self.ncx)
      for i in range(self.M):
        if self.logsca[0]:
          self.C[i] = stats.norm.pdf(self.cx, scale = self.stdc[i] / np.log(self.resd[0]))
        elif self.pola[0]:
          self.C[i] = vmpdf(self.cx, 0., self.stdc[i])
        else:
          self.C[i] = stats.norm.pdf(self.cx, scale = self.stdc[i] / self.resd[0])
        self.C[i] /= unzero(self.C[i].sum())
      self.useKernel(self.C)
      return
    if len(self.stdc) == 1:
      self.stdc = np.tile(self.stdc[0], self.ND)
    # setBins is always called before reach this stage by setData()
    self.calcStds()
    if (self.stdc[0] <= 0.):
      if self.pola[0]:
        self.stdc[0] = self.stdx * float(self.N)**(.6) # a la Bhumbra and Dyball 2010
      else:
        self.stdc[0] = self.stdx * float(self.N)**(-.3) # a la Bhumbra and Dyball 2010
    if not(_ncx):
      if self.logsca[0]:
        _ncx = np.ceil(2. * self.stdw[0] * self.stdc[0] / np.log(self.resd[0]))
      elif self.pola[0]:
        _ncx = np.ceil(0.5 * self.resn[0])
      else:
        _ncx = np.ceil(2. * self.stdw[0] * self.stdc[0] / self.resd[0])
    self.ncx = int(_ncx)
    if self.pola[0]:
      self.cx = vmpdf(np.linspace(-0.5*np.pi, 0.5*np.pi, self.ncx), 0., self.stdc[0])
    else:
      self.cx = stats.norm.pdf(np.linspace(-self.stdw[0], self.stdw[0], self.ncx))
    self.cx /= unzero(self.cx.sum())
    if self.ND < 2:
      self.useKernel(self.cx)
      return
    if (self.stdc[1] <= 0.):
      if self.pola[1]:
        self.stdc[1] = self.stdy * float(self.N)**(.6) # a la Bhumbra and Dyball 2010
      else:
        self.stdc[1] = self.stdy * float(self.N)**(-.3) # a la Bhumbra and Dyball 2010
    if not(_ncy):
      if self.logsca[1]:
        _ncy = np.ceil(2. * self.stdw[1] * self.stdc[1] / np.log(self.resd[1]))
      elif self.pola[1]:
        _ncy = np.ceil(0.5*self.resn[1])
      else:
        _ncy = np.ceil(2. * self.stdw[1] * self.stdc[1] / self.resd[1])
    self.ncy = int(_ncy)
    if self.pola[1]:
      self.cy = vmpdf(np.linspace(-0.5*np.pi, 0.5*np.pi, self.ncy), 0., self.stdc[1])
    else:
      self.cy = stats.norm.pdf(np.linspace(-self.stdw[1], self.stdw[1], self.ncy))
    self.cy /= unzero(self.cx.sum())
    self.C = self.cy.reshape(len(self.cy), 1)*self.cx
    self.C /= unzero(self.C.sum())
    self.useKernel(self.C)
  def setBins(self, _xx = None, _yy = None):
    self.xx, self.yy = _xx, _yy
    self.genBins()
    if self.ncx: self.genKernel(self.ncx, self.ncy) ## need to reconstruct kernel
  def setData(self, _data = None, _C = None):
    if _C is None: _C = []
    if _data is None:
      self.N = 0
      self.M = 0
      self.ND = 0
      self.p = np.empty(0, dtype = float)
      self.P = np.empty( (0,0), dtype = float)
      return
    if type(_data) is np.ndarray:
      self.ND = _data.ndim
      if self.ND == 1:
        self.X = np.copy(_data)
        self.N = len(self.X)
        self.minx = self.X.min()
        self.maxx = self.X.max()
      elif self.ND == 2:
        self.XY = np.copy(_data)
        self.N = self.XY.shape[1]
        self.X = self.XY[0,:]
        self.Y = self.XY[1,:]
        self.minx = self.X.min()
        self.maxx = self.X.max()
        self.miny = self.Y.min()
        self.maxy = self.Y.max()
    elif type(_data) is list:
      self.XX = _data[:]
      self.M = len(self.XX)
      if not(self.M): return
      self.ND = 1
      self.minx = np.inf
      self.maxx = -np.inf
      for data in _data:
        self.minx, self.maxx = min(self.minx, data.min()), max(self.maxx, data.max())
    if self.ND:
      self.setBins(self.xx, self.yy)
      if not(len(_C)):
        if not(self.ncx): self.genKernel()
      else:
        self.useKernel(_C)
  def conv(self, _extents = ['full']):
    if not(self.ND): return None
    if not(type(_extents) is list):
      _extents = [_extents] * self.ND
    elif len(_extents) == 1:
      _extents *= self.ND
    self.extents = _extents
    _extents = self.extents[:]
    for i in range(self.ND):
      if self.pola[i]:
        self.extents[i] = 'wrap'
        _extents[i] = 'full'
      if _extents[i] == 'wrap': _extents[i] = 'full'
    if not(self.M):
      if self.ND == 1:
        if self.pola[0]:
          self.h = np.histogram(cang(self.X), self.xx)[0]
        else:
          self.h = np.histogram(self.X, self.xx)[0]
        self.p = np.fabs(signal.fftconvolve(self.h, self.cx, _extents[0]))
        self.p /= unzero(self.p.sum())
        if self.pola[0] or self.extents[0] == 'wrap':
          self.p = wrapsumarray(self.p, len(self.h))
          self.x = self.xx[:len(self.p)]
        else:
          self.x = extarray(self.xx, len(self.p), self.logsca[0])
        self.nx = len(self.x)
        self.ny = 1
        return self.p
      elif self.ND == 2:
        self.H = np.transpose(np.histogram2d(self.X, self.Y, [self.xx, self.yy])[0]) # transposed histogram (for excellent reasons)
        _P = np.fabs(signal.fftconvolve(self.H, self.C, _extents[0]))
        nm = self.H.shape
        ij = [0, 0]
        if self.pola[0]: ij[1] = nm[1]
        if self.pola[1]: ij[0] = nm[0]
        self.P = wrapsumarray(_P, ij)
        self.P = self.P / unzero(self.P.sum())
        nm = self.P.shape[::-1]
        if self.pola[0]:
          self.x = self.xx[:nm[0]]
        else:
          self.x = extarray(self.xx, nm[0], self.logsca[0])
        if self.pola[1]:
          self.y = self.yy[:nm[1]]
        else:
          self.y = extarray(self.yy, nm[1], self.logsca[1])
        self.nx = len(self.x)
        self.ny = len(self.y)
        return self.P
    else:
      for i in range(self.M):
        if not(i):
          _H = np.histogram(self.XX[i], self.xx)[0]
          self.H = np.empty( (self.M, len(_H)), dtype = float)
          self.H[i, :] = _H
          _P = signal.fftconvolve(_H, self.C[i,:], _extents[0])
          if self.pola[0]:
            _P = wrapsumarray(_P, len(_H))
          self.P = np.empty( (self.M, len(_P)), dtype = float)
          self.P[i, :] = np.fabs(_P)
        else:
          self.H[i, :] = np.histogram(self.XX[i], self.xx)[0]
          _P = np.fabs(signal.fftconvolve(self.H[i, :], self.C[i,:], _extents[0]))
          if self.pola[0]:
            self.P[i, :] = wrapsumarray(_P, len(self.H[i, :]))
          else:
            self.P[i, :] = np.copy(_P)
        self.P[i, :] /= unzero(self.P[i, :].sum())
      if self.pola[0]:
        self.x = self.xx[:self.P.shape[0]]
      else:
        self.x = extarray(self.xx, self.P.shape[1], self.logsca[0])
      self.nx = len(self.x)
      self.ny = 1
      return self.P
  def convolve(self, _data, _C = None, _extents = ['full']):
    if _C is None: _C = []
    self.setData(_data, _C)
    return self.conv(_extents)

#-----------------------------------------------------------------------------------------------------------------------
class censca:
  means = None
  stdvs = None
  def __init__(self, _centrescale = -1): # -1 is off; otherwise axis
    self.gck = gckernel()
    self.setCenSca(_centrescale)
  def setCenSca(self, _centrescale = -1):
    if isint(_centrescale): _centrescale = [_centrescale, _centrescale]
    self.cendim, self.scadim = _centrescale
  def process(self, X, **kwargs):
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
      self.stdvs = unzero(np.std(Y, axis = j, **kwargs))
      Y /= np.tile(self.stdvs.reshape(res), rep)
    return Y


#-----------------------------------------------------------------------------------------------------------------------
class midran:
  maxs = None
  mins = None
  mids = None
  wids = None
  def __init__(self, _middlerange = -1): # -1 is off; otherwise axis
    self.setMidRan(_middlerange)
  def setMidRan(self, _middlerange = -1):
    if isint(_middlerange): _middlerange = [_middlerange, _middlerange]
    self.middim, self.randim = _middlerange
  def process(self, X, **kwargs):
    Y = np.array(X, dtype = float)
    sY = list(Y.shape)
    if len(sY) == 1: sY.append(1)
    oY = [1] * len(sY)
    i, j = self.middim, self.randim
    if i >= 0 or j >= 0:
      self.maxs = Y.max(axis = i, **kwargs)
      self.mins = Y.min(axis = i, **kwargs)
    if i >= 0:
      res = sY[:]; res[i] = 1
      rep = oY[:]; rep[i] = sY[i]
      self.mids = 0.5 * (self.maxs + self.mins)
      Y -= np.tile(self.mids.reshape(res), rep)
    if j >= 0:
      res = sY[:]; res[j] = 1
      rep = oY[:]; rep[j] = sY[j]
      self.wids = unzero(0.5 * (self.maxs - self.mins))
      Y /= np.tile(self.wids.reshape(res), rep)
    return Y

#-----------------------------------------------------------------------------------------------------------------------
class sumndim: # a floating point multidimensional class with independent summating dimensions.
  v = None # parameter vector
  D = None # parameter dimensional changer matrix (2D array)
  n = 0    # number of parameters
  N = 0    # number of dimensions
  def __init__(self, _v = None, _D = None):
    self.setvD(_v, _D)
  def setvD(self, _v = None, _D = None):
    if _v is not None: self.v = np.array(_v, dtype = float)
    if _D is not None: self.D = np.matrix(_D, dtype = float)
    chkvD = True
    if self.v is not None:
      self.n = len(self.v)
    else:
      chkvD = False
    if self.D is not None:
      self.N = len(self.D)
    else:
      chkvD = False
    if not(chkvD): return
    for i in range(self.N):
      if self.n != len(np.ravel(self.D[i])):
        raise ValueError('fpfunc.mutator.setvD(): inputs incommensurate')
  def retvi(self, i = None):
    if i is None: return self.v
    if self.n is None: return None
    if len(i) != self.N:
      raise ValueError('fpfunc.mutator.retvi(): index specification dimension inconsistent with change matrix')
    return np.ravel(self.v + np.array(np.matrix(i, dtype = float) * self.D))

#-----------------------------------------------------------------------------------------------------------------------

class deltaxy (sumndim): # a convenience class of sumndim for two variables over two dimensions
  def __init__(self, *args):
    self.setxy(*args)
  def setxy(self, *_args):
    args = None
    if len(_args) == 1:
      if isarray(_args[0]):
        args = tuple(_args[0])
    if args is None: args = _args
    nargs = len(args)
    self.xy = np.zeros(2, dtype = float)
    MD = None
    XYi = []
    xyi = np.zeros(2, dtype = float)
    for i in range(nargs):
      arg = args[i]
      odd = i % 2
      isd = i > 1
      isa = isarray(arg) and i == nargs - 1
      if not(isa):
        if isd:
          if not(odd):
            xyi = np.zeros(2, dtype = float)
            xyi[0] = float(arg)
          else:
            xyi[1] = float(arg)
            XYi.append(np.copy(xyi))
        else:
          self.xy[i] = float(arg)
      else:
        MD = np.array(arg, dtype = int)
        if isd and not(odd):
          XYi.append(np.copy(xyi))
    self.XY = np.array([xyi]) if not(len(XYi)) else np.array(XYi)
    self.setvD(np.array(self.xy), self.XY)
    if MD is not None: self.setmd(MD)
  def setmd(self, *args): # self.md is entirely cosmetic (unless called by retXY)
    if len(args) == 1:
      if isarray(args[0]):
        self.md = np.array(args[0], dtype = int)
        return
    self.md = np.array(list(args), dtype = int)
  def retxy(self, *args):
    if len(args) == 1:
      if isarray(args[0]):
        return self.retvi(args[0])
    return self.retvi(np.array(list(args), dtype = int))
  def retXY(self, *args):
    self.setmd(*args)
    M = np.prod(self.md)
    X = np.empty(M, dtype = float)
    Y = np.empty(M, dtype = float)
    for i in range(M):
      ind = np.unravel_index(i, self.md)
      X[i], Y[i] = self.retxy(ind)
    return X, Y

#-----------------------------------------------------------------------------------------------------------------------

class logcensca:
  def __init__(self, _logcentrescale = False): # operates only on last dimension of 2D arrays
    self.setLogCenSca(_logcentrescale)
  def setLogCenSca(self, _logcentrescale = False):
    if not(isarray(_logcentrescale)): _logcentrescale = [_logcentrescale]
    if len(_logcentrescale) == 1: _logcentrescale = [_logcentrescale[0], _logcentrescale[0], _logcentrescale[0]]
    self.LCS = _logcentrescale
  def process(self, X):
    Y = np.array(X, dtype = float)
    nY = len(Y)
    lcs = [[]] * 3
    for i in range(3):
      lcs[i] = self.LCS[i]
      if isnumeric(lcs[i]):
        lcs[i] = np.tile(lcs[i], nY)
      if elType(lcs[i]) is int:
        lcsi = np.zeros(nY, dtype = bool)
        lcsi[lcs[i]] = True
        lcs[i] = lcsi
    for i in range(nY):
      lcsi = [lcs[0][i], lcs[1][i], lcs[2][i]]
      for j in range(3):
        if lcsi[j]:
          if j == 0:
            Y[i] = np.log(Y[i])
          elif j == 1:
            Y[i] -= np.mean(Y[i])
          elif j == 2:
            Y[i] /= unzero(np.std(Y[i]))
    return Y
#-----------------------------------------------------------------------------------------------------------------------
