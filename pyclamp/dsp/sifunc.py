import numpy as np
import strfunc
from lsfunc import *

MAXSI = 9223372036854775807

#-------------------------------------------------------------------------------
def intravel(X):
  x = np.array(X.ravel(), dtype = int)
  if x.ndim: 
    return x
  return x.reshape(1)

#-------------------------------------------------------------------------------
def val2uint(x):
  try:
    y = int(x)
  except ValueError:
    return []
  except TypeError:
    return []
  if y < 0: y = range(-y)
  return y

#-------------------------------------------------------------------------------
def list2uint(x):
  y = x[:] if isarray(x) else [x]
  for i in range(len(y)):
    if type(y[i]) is list:
      y[i] = list2uint(y[i])
    else:
      y[i] = val2uint(y[i])
  return y

#-------------------------------------------------------------------------------
def bool2uint(x, arrtype = np.array): # multidimensional
  y = x[:] if isarray(x) else [x]
  n = len(y)
  if not(n):
    return None
  sublist = False
  z = []
  for i in range(n):    
    if isarray(y[i]):
      sublist = True
      z.append(bool2uint(y[i]))
    else:
      if y[i]:
        z.append(i)
  if not(sublist):
    if not(len(z)):
      z = []
    else:
      z = arrtype(z)
    if len(z) == n:
      z = arrtype([-n])     
  return z

#-------------------------------------------------------------------------------
def uint2bool(_x, _n = None): #one-dimensional
  x = np.array([_x]) if Type(_x) is int else np.array(_x, dtype = int)
  n = None
  if len(x) == 1:
    if x[0] < 0: # this trumps _n
      n = -x[0]
      x = np.arange(n)
  if n is None:
    n = _n if _n is not None else np.max(x) + 1
  y = np.zeros(n, dtype = bool)
  y[x] = True
  return y
  
#-------------------------------------------------------------------------------
def longest(x): # returns longest contiguous index range of True from the boolean array x
  nx = x.size
  i = np.nonzero(np.logical_not(x))[0]
  if i.size == 0:
    return np.arange(nx, dtype = int)
  elif i.size == 1:
    if float(i) < 0.5*float(nx):
      return np.arange(i, nx)
    else:
      return np.arange(i)
  i = np.hstack( (0, i, nx) )
  d = np.diff(i)
  d = np.argmax(d)
  return np.arange(i[d], i[d+1]) 
  
#-------------------------------------------------------------------------------
def cont(x):
  nx = x.size
  if nx == 0: 
    return np.empty(0, dtype = int)
  elif nx == 1:
    return np.arange(nx, dtype = int)
  dx = np.hstack( (1, np.diff(x)) )
  return longest(dx == 1)

#-------------------------------------------------------------------------------
def int2bool(_x, minlength = 1):
  x = int(_x)
  if x < 0:
    raise ValueError('Cannot convert negative integers to binary')
  y = np.zeros(0, dtype = bool)
  while (x):
    y = np.hstack( (bool(x % 2), y) )
    x /= 2
  if len(y) < minlength:
    y = np.hstack( (np.zeros(minlength-len(y), dtype = bool), y))
  return y
  
#-------------------------------------------------------------------------------
def rarange(n, i0 = None, i1 = None, i2 = None):
  if i0 is None: return np.arange(n)
  x = np.arange(i0, i1, i2)
  y = np.empty(n, dtype = x.dtype)
  m = len(x)
  i = 0
  j = m
  while j < n:
    y[i:j] = x
    i = j
    j += m
  if i < n:
    m = n - i
    y[i:] = x[:m]
  return y 

#-------------------------------------------------------------------------------
def factorial(x):
  x = np.arange(1,x+1, dtype = float)
  if not(len(x)): return 0
  return int(np.round(np.prod(x)))

#-------------------------------------------------------------------------------
def omagic(_n, forcematrix = False, **kwds):
  n = float(_n)
  p = np.mat(np.arange(n)) + 1.
  nm = n*np.mod(p.T + p-0.5*(n+3.), n)
  m = np.mod(p.T + 2.*p-2., n)
  M = np.mat(nm + m + 1., dtype = int)
  if forcematrix: return np.mat(M, **kwds)
  return np.array(M, **kwds)

#-------------------------------------------------------------------------------
def emagic(_n, forcematrix = False, **kwds):
  n = int(_n)
  p = n / 2
  p2 = p**2
  M = omagic(p, True)
  M = np.vstack( (np.hstack( (M, M+2*p2) ), np.hstack( (M+3*p2, M+p2) )) )
  if n == 2: 
    if forcematrix: return np.mat(M, **kwds)
    return np.array(M, **kwds)
  i = np.mat(np.arange(p)).T
  k = (n-2)/4
  j = np.hstack( (np.mat(np.arange(k)), np.mat(np.arange(n-k+1, n))) )
  M[np.vstack((i, i+p)), j] = M[np.vstack((i+p, i)), j]
  i = k + 1
  j = np.hstack( (0, i) )
  M[np.vstack((i, i+p)), j] = M[np.vstack((i+p, i)), j]
  if forcematrix: return np.mat(M, **kwds)
  return np.array(M, **kwds)

#-------------------------------------------------------------------------------
def eemagic(_n, forcematrix = False, **kwds):
  n = int(_n)
  J = np.mat(0.5*np.mat(np.mod(np.arange(n)+1, 4), dtype = float), dtype = int)
  K = np.mat(J.T == J, dtype = int)
  nn = n*n
  M = np.mat(np.arange(1, nn, n, dtype = int)).T + np.mat(np.arange(n), dtype = int)
  m = np.multiply(K, (nn+1 - M*2))
#  M += m
  if forcematrix: return np.mat(M, **kwds)
  return np.array(M, **kwds)

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

#-------------------------------------------------------------------------------
def magic(_n, forcematrix = False, **kwds):
  n = int(_n)
  if np.mod(n, 2): return omagic(n, forcematrix, **kwds)
  if np.mod(n, 4): return emagic(n, forcematrix, **kwds)
  return eemagic(n, forcematrix, **kwds)

#-------------------------------------------------------------------------------
def magiperm(_n, transpose = True):
  n = int(_n)
  s = np.ceil(np.sqrt(float(n)))
  S = magic(s)
  if transpose: S = S.T
  m = np.ravel(S) - 1
  return m[m < n]

#-------------------------------------------------------------------------------
def intspace(_lo = 0, _hi = 9, _n = 10, geoscale = False):
  lo, hi, n = int(_lo), int(_hi), int(_n)
  if not(geoscale): return np.linspace(lo, hi, n, dtype = int)
  if n == 1: return np.array([lo], dtype = int)
  if n == 2: return np.array([lo, hi], dype = int)
  if lo == hi: return np.tile(lo, n)
  slo, shi = np.sign(lo), np.sign(hi)
  if not(slo): 
    slo = shi
  elif not(shi):
    shi = slo
  if slo != shi: raise ValueError("Limits of incompatible polarities")
  alo, ahi = int(np.abs(lo)), int(np.abs(hi))
  reverse = alo > ahi
  if reverse: alo, ahi = ahi, alo
  lhi = np.log(ahi)
  i = max(1, alo)
  done = i >= ahi
  m = float(n - 1)
  while not(done):
    j = i + 1
    fi, fj = float(i), float(j)
    li = np.log(fi)
    dd = (lhi - li) / m
    if np.exp(li+dd) >= fj:
      done = True
    else:
      i = j
      done = i >= ahi
      if done: i = alo
  i = min(i, ahi)
  li = np.log(i)
  x = np.array(np.round(np.exp(np.linspace(li, lhi, n))), dtype = int)
  if slo < 0: x = -x
  if reverse: x = x[::-1]
  return x

#-------------------------------------------------------------------------------
def replicind(irep = 1, maxi = 10, orep = 1): # for constructing index array with replications
  return np.tile(np.ravel(np.tile(np.arange(maxi, dtype = int).reshape(maxi, 1), (1, irep))), (1, orep))[0]

