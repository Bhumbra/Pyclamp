# A data types modules

import numpy as np

_SI_MAX_ = 9223372036854775807
_FP_MAX_ = 1.7976931348623157e308

#-------------------------------------------------------------------------------
def Type(x):
  y = type(x)
  if y == np.float64:
    return float
  elif y == np.int64:
    return int  
  elif y == np.bool_:
    return bool
  else:
    return y
#-------------------------------------------------------------------------------
def isarray(x):
  if type(x) is list:
    return True
  if type(x) is tuple:
    return True
  if type(x) is np.ndarray:
    return True
  if type(x) is np.matrixlib.defmatrix.matrix:
    return True
  return False

#-------------------------------------------------------------------------------
def eltype(x, typefunc = type):
  if type(x) is np.matrixlib.defmatrix.matrix:
    xs = list(x.shape)
    if xs[0] and xs[1]:
      return typefunc(x[0,0], typefunc)
    else:
      return None
  if isarray(x):
    if len(x):
      return eltype(x[0], typefunc)
  return typefunc(x)

#-------------------------------------------------------------------------------
def elType(x):
  return eltype(x, Type)

#-------------------------------------------------------------------------------
def isbool(x):
  if type(x) is bool or type(x) is np.bool_: return True
  return False

#-------------------------------------------------------------------------------
def isint(x):
  if type(x) is int: return True
  if type(x) is np.int32 or type(x) is np.int64: return True
  return False

#-------------------------------------------------------------------------------
def isfloat(x):
  if type(x) is float: return True
  if type(x) is np.float32 or type(x) is np.float64: return True
  return False

#-------------------------------------------------------------------------------
def isnum(x):
  if isbool(x): return True
  if isint(x): return True
  if isfloat(x): return True
  return False

#-------------------------------------------------------------------------------
def isempty(x):
  if isint(x):
    return False
  if isfloat(x):
    return False
  if len(x):
    return False
  return True

#-------------------------------------------------------------------------------
def isnumeric(_x): # is float-convertible
  if type(_x) is str:
    x = _x.replace(' ', '').lower()
    if x == 'false' or x == 'true': return True
  try:
    float(_x)
  except (ValueError, TypeError) as MultiError:
    MultiError = False
    return MultiError
  else:
    return True

#-------------------------------------------------------------------------------
def isNumeric(_X, retAll = False):
  n = len(_X)
  y = np.empty(n, dtype = bool)
  for i in range(n):
    y[i] = isnumeric(_X[i])
  Y = np.all(y)
  if retAll: return Y, y
  return Y

#-------------------------------------------------------------------------------
def isNaN(x):
  try:
    a = np.isnan(x)
    return a
  except TypeError:
    return True

#-------------------------------------------------------------------------------
def notnan(x):
  if (isnum(x)):
    return np.logical_not(isNaN(x))
  return True

#-------------------------------------------------------------------------------
def argtrue(x):
  i = np.nonzero(x)
  if len(i) == 1: 
    i = i[0]
    if len(i): return i
    return ()
  return i

#-------------------------------------------------------------------------------
def nparray(x, **kwds):
  if isarray(x): return np.array(x, **kwds)
  return np.array([x], **kwds)

#-------------------------------------------------------------------------------
def nparray2D(x, **kwds):
  if isarray(x): 
    Y = np.array(x, **kwds)
  else:
    Y = np.array([x], **kwds)
  if np.ndim(Y) > 2: Y = np.ravel(Y)
  while np.ndim(Y) < 2: Y = np.array([Y])
  return Y
#-------------------------------------------------------------------------------
