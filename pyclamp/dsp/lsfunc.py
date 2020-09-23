import numpy as np
import os
from pyclamp.dsp.strfunc import *
from pyclamp.dsp.dtypes import *

def listcurdir():
  return os.listdir(os.curdir)

def copyarray(x):
  if not(isarray(x)):
    raise ValueError("Cannot copy an array when input is not an array")
  if type(x) is np.ndarray:
    return np.copy(x)
  return x[:]  

def nDim (x):
  y = x
  n = 0
  while (type(y) is list or type(y) is np.ndarray):
    try:
      if len(y):
        n += 1
        y = y[0]
      else:  
        n -= 1
        y = None
    except TypeError:
      y = None
  return n  

def lDim (x):
  n = nDim(x)
  d = [0] * n
  y = x
  for i in range(n):
    d[i] = len(y)
    if i < n - 1:
      y = y[0]
  return d

def lel0(x, d):
  i = 0  
  y = x
  while i <= d:
    y = y[0]

def flattenlist(X):
  y = []
  for x in X:
    if type(x) is list:
      y += flattenlist(x)
    else:
      y += [x]
  return y    

def flattenarray(X, depth = 0, currentdepth = 0):
  if not(isarray(X)): return X
  if isint(X) or isfloat(X): return X
  currentdepth += 1
  flatten = currentdepth >= depth
  Y = [] if flatten else X[:]
  n = len(X)
  for i in range(n):
    x = X[i]
    xarr = isarray(x)
    if flatten:
      if xarr:
        Y += flattenarray(list(x), depth, currentdepth)
      else:
        Y += [x]
    elif xarr:
      Y[i] = flattenarray(list(x), depth, currentdepth)
  return Y
    
def nLen(x):
    if type(x) is list:
        return len(x)
    elif type(x) is float or type(x) is int:
        return 1
    elif type(x) is np.ndarray:
        return x.shape[0]
    else:
        return 1

def nRows(x):
    if type(x) is list:
        return len(x)
    if x.ndim == 1:
        return 1
    else:
        return x.shape[0]
    
def nCols(x):
    if type(x) is list:
        return len(x[0]) 
    if x.ndim == 1:
        return x.shape[0]
    else:
        return x.shape[1]    
    
def repl(x, n):
    if type(x) is list:
        if len(x) == n:
            return x[:]
        else:
            return [x[:]] * n
    else:
        return [x] * n     

def rep2(x, y):
  ndx = nDim(x)
  ndy = nDim(y)
  dx = lDim(x)
  dy = lDim(y)
  if ndx == ndy: 
    iden = 1
    for i in range (ndx):
      if dx[i] != dy[i]: iden = 0
    if iden: return x, y
  else:  
    dn = abs(ndy - ndx)
  X = x[:] if type(x) is list else [x]
  Y = y[:] if type(y) is list else [y]
  if ndx < ndy:    
    for i in range(ndy-1, ndy-dn-1, -1):
      X = [X] * dy[i];
  elif ndx > ndy:
    for i in range(ndx-1, ndx-dn-1, -1):
      Y = [Y] * dx[i];
  else:
    for i in range(ndx):
      if dx[i] > dy[i] and dy[i] == 1:
        for j in range(dy[i-1]):
          l0 = lel0(Y, j)
          if len(l0) == 1:
            l0 *= dx[i]
      elif dy[i] > dx[i] and dx[i] == 1:
        for j in range(dx[i-1]):
          l0 = lel0(Y, j)
          if len(l0) == 1:
            l0 *= dy[i]   
  return X, Y       

def unique(X):
  if type(X) is np.ndarray:
    return np.unique(X)
  if type(X) is tuple:
    return tuple(set(x))
  return list(set(x))

def listfind(ref, val, defn = None):
  try:
    ind = ref.index(val)
  except ValueError:
    ind = defn 
  return ind

def argfind(ref, val, defn = None):
  if type(ref) is not list: ref = list(ref)
  if type(val) is not list: return listfind(ref, val, defn)
  n = len(val)
  ind = [None] * n
  for i in range(n):
    ind[i] = listfind(ref, val[i], defn)
  return ind 

def extnum(X):
  if not(isarray(X)):
    if isnum(X):
      return X
    else:
      return None
  N = nDim(X)  
  if N == 0:
    return None
  Y = None
  if N == 1:
    for x in X:
      if isnum(x):
        if Y is None:
          Y = [x]
        else:
          Y.append(x)
    return Y    
  for x in X:
    y = extnum(x)
    if y is not None:
      if Y is None:
        Y = [y]
      else:
        Y.append(y)
  return Y

def extnotnan(X):
  if not(isarray(X)):
    if isnum(X):
      return X
    else:
      return None
  N = nDim(X)  
  if N == 0:
    return None
  Y = None
  if N == 1:
    for x in X:
      if notnan(x):
        if Y is None:
          Y = [x]
        else:
          Y.append(x)
    return Y    
  for x in X:
    y = extnotnan(x)
    if y is not None:
      if Y is None:
        Y = [y]
      else:
        Y.append(y)
  return Y


def listmat(nr, nc):
  x = list(range(nr*nc))
  y = []
  for i in range(nr):
    y.append(x[i*nc:(i+1)*nc])
  return y    

def listcomplete(_x, ref = None):
  if ref is None: ref = []
  x = _x[:]
  i = len(x)
  n = len(ref)
  while i < n:
    x.append(ref[i])
    i+=1
  return x;  

def cosad(list0, list1 = None): # copy or sum absolute differences
  if list1 is None:
    return [np.copy(lis) for lis in list0]
  nlist = len(list0)
  list2 = np.array(nlist, dtype = float)
  for i in range(len(list0)):
    list2[i] = np.fabs(list0 - list1).sum()
  return list2

def list2array(x):
  n = len(x)
  y = [[]] * n
  for i in range(n):
    xi = x[i]
    if type(xi) is list:
      y[i] = list2array(xi)
    elif isnumeric(x[i]):
      y[i] = float(x[i])
    else:
      y[i] = np.nan
  try:
    Y = np.array(y, dtype = float)
  except ValueError:
    Y = y 
  return Y

def array2list(x):
  if type(x) is np.ndarray:
    return x.tolist()
  else:
    return list(x)
  
def nanarray2list(_X):
  X = []
  m = nDim(_X)
  if m > 1:
    if type(_X) is np.ndarray:
      X = _X.tolist()    
    else:
      X = _X[:]  
  elif m == 1:
    if type(_X) is list:
      X = np.array(_X, dtype = float) 
    return X[np.nonzero(np.logical_not(np.isnan(X)))[0]]
  n = len(_X)  
  for i in range(n):
    X[i] = nanarray2list(X[i])
  return X

def repl2dim(_X, _dims):
  X, dims = np.array(_X), np.array(_dims)
  sX = X.shape
  nX, nD = len(sX), len(dims)
  m = min(nX, nD)
  n = max(nX, nD)
  res = np.ones(n, dtype = int)
  rep = np.ones(n, dtype = int)
  for i in range(n):
    if i < m:
      rep[i] = int(np.floor(dims[i] / sX[i]))
      res[i] = int(np.floor(dims[i] / rep[i]))
      if rep[i] * res[i] < dims[i]:
        raise ValueError("Incompatible dimensions of inputs.")
    elif i<nD:
      rep[i] = dims[i]
    elif i<nX:
      res[i] = sX[i]
  return np.tile(X.reshape(res), rep)

def numstr2list(_x, _allowsing = None):
  defallowsing = [False, True]
  if not(len(_x)): return []
  if _allowsing is None:
    _allowsing = defallowsing[:]
  allowsing = _allowsing[:] if type(_allowsing) is list else [_allowsing]
  if len(allowsing) == 0: allowsing = defallowsing[:]
  if type(allowsing[0]) is bool: allowsing.insert(0, 0)
  allowsing[0] += 1
  x = _x[:]
  X = []
  done = False
  nx = len(x)
  h = 0
  i = None
  j = None
  while not(done):
    xh = x[h]
    if xh == '[' or xh == '(' or xh == '{':
      i = h + 1
      j = bracematch(x, h)
      if j == -1: raise ValueError('Unmatched braces.')
      X.append(numstr2list(x[i:j], allowsing))
      h = j
      i = None
      j = None
    elif isnumeric(xh) or xh == '.' or xh == '-' or xh == 'e':
      if i is None:
        i = h
      j = h
      if h == nx - 1:
        j = h + 1
        X.append(str2num(x[i:j]))
    elif i is not None:
      j = h
      X.append(str2num(x[i:j]))
      i = None
      j = None
    h += 1
    done = h == nx
  a = allowsing[-1] if allowsing[0] >= len(allowsing) else allowsing[allowsing[0]]
  if not(a) and len(X) == 1: X = X[0]  
  return X 

def listtranspose(X):
  if nDim(X) != 2:
    raise ValueError("Transponse only possible in two dimensions")
  nCols = len(X)
  nrows = np.zeros(nCols,  dtype = int)
  for i in range(nCols):
    nrows[i] = len(X[i])
  nRows = np.max(nrows)
  Y = [[]] * nRows
  for j in range(nRows):
    Y[j] = [[]] * nCols
    for i in range(nCols):
      if j < nrows[i]:
        Y[j][i] = X[i][j]
      else:
        del Y[j][i]
  return Y      

def xy2list(x, y, xlbl = None, ylbl = None):
  n, _n = len(x), len(y)
  if n != _n: raise ValueError("x and y inputs incommensurate")
  if xlbl is not None and ylbl is None: ylbl = ''
  if xlbl is None and ylbl is not None: xlbl = ''
  k = 0
  if xlbl is None:
    XY = [None] * n
  else:
    XY = [None] * (n + 1)
    XY[0] = [xlbl, ylbl]
    k += 1
  for i in range(n):
    XY[k] = [x[i], y[i]]
    k += 1
  return XY

def xY2list(x, Y, xlbl = None, Ylbl = None):
  n, N = len(x), len(Y)
  M = N + 1
  for y in Y:
    if n != len(y): raise ValueError("x and Y inputs incommensurate")
  if xlbl is not None and Ylbl is None:
    Ylbl = range(N)
  if xlbl is None and Ylbl is not None: xlbl = ''
  k = 0
  if xlbl is None:
    XY = [None] * n
  else:
    XY = [None] * (n + 1)
    XY[0] = [None] * M
    XY[0][0] = xlbl
    for j in range(N):
      XY[0][j+1] =  Ylbl[j]
    k += 1
  for i in range(n):
    XY[k] = [None] * M
    XY[k][0] = x[i]
    for j in range(N):
      XY[k][j+1] = Y[j][i]
    k += 1
  return XY

class listtable:
  X = None
  data = None
  n = 0
  m = 0
  defkeycols = []
  deffldrows = 0 
  def __init__(self, _X = None, _fldrows = None, _keycols = None):
    self.flds = []
    self.keys = []
    self.initialise(_X, _fldrows, _keycols)
  def initialise(self, _X = None, _fldrows = None, _keycols = None):
    if _fldrows is None: _fldrows = self.deffldrows
    if _keycols is None: _keycols = self.defkeycols
    self.setSpec(_fldrows, _keycols)
    self.setList(_X)
  def setSpec(self, _fldrows = None, _keycols = None):
    if _fldrows is not None: 
      if isint(_fldrows): _fldrows = [_fldrows]
      self.fldRows = _fldrows[:]
      self.nflds = len(self.fldRows)
      self.mflds = None
      for k in range(self.nflds):
        self.mflds = self.fldRows[k] if self.mflds is None else max(self.mflds, self.fldRows[k])
    if _keycols is not None:
      if isint(_keycols): _keycols = [_keycols]
      self.keyCols = _keycols[:]
      self.nkeys = len(self.keyCols)
      self.mkeys = None
      for k in range(self.nkeys):
        self.mkeys = self.keyCols[k] if self.mkeys is None else max(self.mkeys, self.keyCols[k])
    if self.X is not None: self.setList()  
  def setList(self, _X = None):
    if _X is not None: self.X = _X
    if self.X is None: return
    self.nrows = len(self.X)
    self.ncols = [[]] * self.nrows
    for i in range(self.nrows):
      self.ncols[i] = len(self.X[i])
    self.readKeys()
    self.readFlds()
    self.readData()
  def readKeys(self):
    self.v = []
    self.keys = []
    if self.nkeys == 0:
      for i in range(self.nrows):
        ok = True
        for k in range(self.nflds):
          if i == self.fldRows[k]: ok = False
        if ok:
          self.v.append(i)
    else:
      for i in range(self.nrows):
        Xi = self.X[i]
        ok = self.mkeys < self.ncols[i]
        if ok:
          for k in range(self.nflds):
            if i == self.fldRows[k]: ok = False
        if ok:
          keyi = ''
          for k in range(self.nkeys):
            Xik = Xi[self.keyCols[k]]
            if isempty(Xik):
              ok = False
            else:
              keyi = ''.join((keyi, str(Xik)))
        if ok:
          self.v.append(i)
          self.keys.append(keyi[:])
    self.n = len(self.v)
    return self.keys
  def readFlds(self):
    self.maxf = None
    for k in range(self.nflds):
      i = self.fldRows[k]
      self.maxf = self.ncols[i] if self.maxf is None else max(self.maxf, self.ncols[i])      
    self.f = []
    self.flds = []
    for j in range(self.maxf):
      ok = True
      fldj = ''
      for k in range(self.nflds):
        i = self.fldRows[k]
        Xij = self.X[i][j]
        if isempty(Xij):
          ok = False
        else:
          fldj = ''.join((fldj, str(Xij)))
      if ok:
        self.f.append(j)
        self.flds.append(fldj[:])
    self.m = len(self.flds)
    return self.flds
  def readData(self):
    self.data = [[]] * self.n
    for i in range(self.n):
      rowi = self.v[i]
      self.data[i] = [''] * self.m
      for j in range(self.m):
        colj = self.f[j]
        if colj < self.ncols[rowi]:
          self.data[i][j] = self.X[rowi][colj]
    return self.data
  def setData(self, _data = None, _flds = None, _keycols = None):
    if _data is None: _data = self.data
    if _flds is None: _flds = self.flds
    if _keycols is None: _keycols = self.keyCols
    self.n = len(_data)
    self.m = len(_flds)
    X = [[]] * (self.n + 1)
    X[0] = [[]] * self.m
    for j in range(self.m):
      X[0][j] = _flds[j]
    for i in range(self.n):
      X[i+1] = _data[i][:]
    self.initialise(X, 0, _keycols) 
    return self.X
  def findFlds(self, _fld):
    if isint(_fld):
      fld = _fld
    elif type(_fld) is str:
      fld = listfind(self.flds, _fld)
    else:
      if len(_fld) != self.nflds:
        raise ValueError("Input argument incommensurate with field specification")
      fld = ''
      for __fld in _fld:
        fld = ''.join((fld, __fld))
      fld = listfind(self.flds, fld)
    return fld
  def retFlds(self, _fld, **kwargs):
    if type(_fld) is list:
      nfld = len(_fld)
      Fld = [[]] * nfld
      for i in range(nfld):
        Fld[i] = self.retFlds(_fld[i], **kwargs)
      return Fld
    fld = self.findFlds(_fld)
    if fld is None: return None
    x = [None] * self.n
    for i in range(self.n):
      if fld < len(self.data[i]):
        x[i] = self.data[i][fld]
    if not(len(kwargs)): return x 
    havenan = False
    for i in range(len(x)):
      setnan = not(isnumeric(x[i]))
      if not(setnan):
        if type(x[i]) is str:
          x[i] = x[i].lower()
          if x[i] == 'false':
            x[i] = '0'
          elif x[i] == 'true':
            x[i] = '1'
      else:
        x[i] = 'NaN'
        havenan = True
    if havenan and kwargs['dtype'] is not float:
      print("listtable.retFlds Warning: non-numeric data type encountered during non-float conversion")
    x = np.array(x, dtype = float) 
    return  np.array(x, **kwargs)
  def copy(self):
    return listtable(self.X, self.fldRows, self.keyCols)
  def findKeys(self, _keys):
    if type(_keys) is not list: _keys = [_keys]
    I = [[]] * len(_keys)
    for i in range(len(_keys)):
      I[i] = listfind(self.keys, _keys[i])
    return I
  def deleteByIndex(self, I = None):
    if type(I) is not list:
      if not(isint(I)) and I is not None: ValueError('Entry deletion by index requires integer (list) as input.')    
      I = [I]
    I = sorted(I)[::-1]  
    for i in I:
      if i is not None:
        if i>=0 and i<self.n: 
          self.data.pop(i)
          self.keys.pop(i)
    dn = self.n - len(self.keys)
    self.n = len(self.keys)
    return dn
  def deleteByKey(self, _keys):
    I = self.findKeys(_keys)
    return self.deleteByIndex(I)
  def sortByKeys(self, **kwArgs):
    I = np.argsort(self.keys, **kwArgs)
    _keys = [[]] * self.n
    _data = [[]] * self.n
    _ncols = [[]] * self.n
    for i in range(self.n):
      k = I[i]
      _keys[i] = self.keys[k]
      _data[i] = self.data[k]
      _ncols[i] = self.ncols[k]
    self.keys = _keys
    self.data = _data
    self.ncols = _ncols
  def join(self, other):
    if self.data is None or other.data is None:
      raise ValueError("No data read in one of the source classes.")
    if self.nkeys < 1 or other.nkeys < 1:
      raise ValueError("Key definitions mandatory in both sources classes")
    obj = self.copy()
    I = obj.findKeys(other.keys)
    for k in range(other.n):
      i = I[k]
      if i is not None:
        for j in range(obj.ncols[i], obj.m):
          obj.data[i].append(None)
        for j in range(other.ncols[k]):
          obj.data[i].append(other.data[k][j])
        obj.ncols[i] = len(obj.data[i])
    for fld in other.flds:
      obj.flds.append(fld)
    obj.m = len(obj.flds)        
    return obj
  def union(self, other, keepcurr = True): #keepold = False overwrites existing key data
    if self.data is None or other.data is None:
      raise ValueError("No data read in one of the source classes.")
    if self.m != other.m:
      raise ValueError("Fields must match")
    J = [[]] * self.m
    for i in range(self.m):
      J[i] = other.findFlds(self.flds[i])
      if J[i] < 0: raise ValueError("Fields must match")
    obj = self.copy()
    for i in range(len(other.data)):
      k = obj.findKeys(other.keys[i]) if len(self.keyCols) else -1
      if k < 0:
        obj.data.append([[]] * self.m)
        for j in range(self.m):
          obj.data[-1][j] =  other.data[i][J[j]]
      elif not(keepcurr):
        for j in range(self.m):
          obj.data[k][j] =  other.data[i][J[j]]
    obj.setData(obj.data, obj.flds, obj.keyCols)
    return obj
  def filterEntries(self, _spec = None): 
    if _spec is None: return
    spec = np.array(_spec, dtype = bool)
    if len(spec) != self.n:
      raise ValueError("Filter specification dimensions incommensurate with data")
    _n = self.n
    self.n = np.sum(spec)
    _data = [[]] * self.n
    _keys = [[]] * self.n
    _ncols = [[]] * self.n
    j = 0
    for i in range(_n):
      if spec[i]:
        _data[j] = self.data[i]
        _keys[j] = self.keys[i]
        _ncols[j] = self.ncols[i]
        j += 1
    self.data = _data
    self.keys = _keys
    self.ncols = _ncols
    return _n - self.n # returns number filtered
  def addFlds(self, _flds = None, _data = None, _indx = None): # Note this has no effect on keys
    if type(_flds) is not list: _flds = [_flds]
    if type(_indx) is not list: _indx = [_indx]
    nf = max(len(_flds), len(_indx))
    if nDim(_data) == 2: nf = max(nf, len(_data)) 
    if len(_flds) < nf:
      if len(_flds) > 1:
        raise ValueError('Field/data/index dimensions incommensurate')
      else:
        _flds *= nf
    if len(_indx) < nf:
      if len(_indx) > 1:
        raise ValueError('Field/data.index dimensions incommensurate')
      else:
        _indx *= nf
    if nDim(_data) == 0: _data = [_data]
    if nDim(_data) == 1: _data = [_data]
    if nf > 1 and len(_data) != nf:
      _data *= nf
    for i in range(nf):
      if _indx[i] is None:
        _indx[i] = self.m
      if _flds[i] is None:
        _flds[i] = "Field_" + str(self.m+i)
    J = np.argsort(_indx)
    for f in range(nf):
      j = J[f]
      fld = _flds[j]
      ind = _indx[j] + f
      self.flds.insert(ind, fld)
    self.m = len(self.flds)
    for i in range(self.n):
      for f in range(nf):
        j = J[f]
        ind = _indx[j] + f
        if ind > len(self.data[i]):
          self.data[i].append([None]*(ind-len(self.data[i])))
        self.data[i].insert(ind, _data[j][i])
        self.ncols[i] = len(self.data[i])
    self.setData()
     
