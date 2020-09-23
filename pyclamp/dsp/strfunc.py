# string functions

import numpy as np
from pyclamp.dsp.dtypes import *

#-------------------------------------------------------------------------------
def strfind(S, s):
  I = []
  d = len(s)
  done = len(S) == 0 or d == 0
  i = 0
  while not(done):
    k = S[i:].find(s)
    done = k < 0
    if not(done):
      i += k
      I.append(i)
      i += d
  return I

def str2num(_s):
  s = _s.replace(' ', '')
  if s == "True": return True
  if s == "False": return False
  if s.find('.') == -1 and s.find('e') == -1:
    return int(s)
  else:
    return float(s)

def str2var(s):
  if isnumeric(s):
    return str2num(s)
  return s

def str2bool(_s):
  s = _s.replace(' ', '')
  if s.lower() == 'false': return False
  if s == '0': return False
  return True

def str2ind(_S):
  S = _S.lower()
  m = 1
  M = 0
  for i in range(len(S)-1, -1, -1):
    I = ord(S[i]) -96
    M += I*m
    m *= 26
  return M - 1

def str2rci(S):
  s = ''
  n = ''
  for i in range(len(S)):
    Si = S[i]
    if isnumeric(Si):
      n = ''.join((n, Si))
    else:
      s = ''.join((s, Si))
  c = str2ind(s) if len(s) else -1
  r = int(n) if len(n) else -1
  return r, c

def bracematch(s, i):
  n = len(s)
  if i < 0 or i >= n: return -1
  si = s[i]
  sj = None
  d = 1
  if si == '[': sj = ']'
  if si == '{': sj = '}'
  if si == '(': sj = ')'
  if si == ']': sj = '['; d = -1
  if si == '}': sj = '{'; d = -1
  if si == ')': sj = '('; d = -1
  if sj is None: return -1
  j = i
  done = False
  count = 1
  while not(done):
    j += d
    if j < 0 or j == n: return -1
    sd = s[j]
    if sd == si: 
      count += 1
    elif sd == sj:
      count -= 1
      if not(count):
        done = True
  return j

def int2str(_x, n = 0):
  x = int(_x)
  s = '-' if x < 0 else ''
  y = str(abs(x))
  m = len(y)
  z = '' if m >= n else '0'*(n-m)
  return s + z + y

def repq(x, c = ' ', _q = ["'", '"'], s = None): # substitutes character inside single qualifiers 
  if type(_q) is str: _q = (_q)
  if s is None: s = chr(26)
  q = list(_q)
  nq = len(q)
  iq = [0] * nq
  nx = len(x)
  X = list(x)
  for i in range(nx):
    xi = x[i]
    Q = False
    for j in range(nq):
      if xi == q[j]:
        Q = True
        iq[j] += 1
    if not(Q) and xi == c:
      for j in range(nq):
        if iq[j] % 2:
          X[i] = s
  return "".join(X)

def repb(x, c = ',', _b = ['(', '['], s = None): # substitutes character inside double qualifiers 
  if type(_b) is str: _b = (_b)
  if s is None: s = chr(26)
  b = list(_b)
  nb = len(b)
  nx = len(x)
  X = list(x)
  done = nx == 0
  i = 0
  while not(done):
    xi = x[i:]
    il = -1
    for j in range(nb):
      jl = xi.find(b[j])
      if jl >= 0:
        il = jl if il < 0 else min(il, jl)
    done = il < 0
    if not(done):
      ir = bracematch(xi, il)
      if ir >= 0:
        for h in range(il, ir):
          k = i + h
          if X[k] == c:
            X[k] = s
        i += ir
      else:
        i = len(x)
  return ''.join(X)

def rmchr(x, c = ' ', q = ["'", '"'], s = None): # remove character unless within qualifier
  if s is None: s = chr(26)
  xq = repq(x, c, q, s)
  yq = xq.replace(c, '')
  y = yq.replace(s, c)
  return y

def parsevarstr(x):
  if isnumeric(x): return str2num(x)
  if len(x) > 2:
    if x[0] == '"' and x[-1] == '"': return x[1:-1]
    if x[0] == "'" and x[-1] == "'": return x[1:-1]
  return x 

def parseargstr(_x, d = ',', rec = False): # parses string/numeric argument list/tuple with delimiter d
  x = _x if rec else rmchr(_x)
  lx = len(x)
  s = chr(26)
  if not(len(x)): return None
  Con, Tup = False, False
  if x[0] == '[' and x[-1] == ']':
    x = x[1:-1]
    Con = True
  elif x[0] == '(' and x[-1] == ')':
    x = x[1:-1]
    Tup = True
    Con = True
  if not(len(x)): # deal with trivial case of [] and ()
    if Con:
      if Tup: return ()
      return []
    return None
  if not(Con):
    return parsevarstr(x)
  ir, iR = -1, -1
  il = x.find('(')
  iL = x.find('[')
  if il < 0 and iL < 0: # no internal braces
    xq = repq(x, d) # substitute qualified delimiters
    X = xq.split(d)
    n = len(X)
    if n == 1:
      X = parsevarstr(X[0].replace(s, d))
      ''' single element tuples not an option
      if Tup: return (X)
      if Con: return [X]
      '''
      if Con: 
        X = [X]
        if Tup: return tuple(X)
        return X
      return X
    for i in range(n):
      X[i] = X[i].replace(s, d)
      X[i] = parsevarstr(X[i])
    if Tup: return tuple(X)
    return X

  # Protect braced delimiters before recursive interpretation
  xb = repb(x)
  X = xb.split(d)
  n = len(X)
  if n == 1:
    X = parseargstr(X[0].replace(s, d), d, True)
    ''' # single element tuples not an option...
    if Tup: return (X)
    if Con: return [X]
    '''
    if Con: 
      X = [X]
      if Tup: return tuple(X)
      return X
    return X
  for i in range(n):
    X[i] = parseargstr(X[i].replace(s, d), d, True)
  if Tup: return tuple(X)
  return X

def parseexpstr(_x, d = ','): # parses something along the lines of function(arg1, arg2, ...)
  x = rmchr(_x)
  il = _x.find('(')
  ir = bracematch(_x, il)
  if il < 0 or ir < 0: return [x, ()]
  return [x[:il], parseargstr(x[il:ir+1], d, True)]

def uintparseunique(x, n = None):
  x = x.replace(' ', '')
  if not(len(x)): 
    if n is not None: 
      return np.arange(n).tolist()
    else:
      return []
  X = parseargstr('['+x+']')
  I = np.empty(0, dtype = int)
  for x in X:
    if type(x) is str:
      if len(x):
        if x[-1] == '-':
          if n is None:
            raise ValueError("Without maximum length specification, cannot parse: " + x)
          else:
            I = np.hstack((I, np.arange(int(x[:-1]), n, dtype = int)))
        else:
          i = x.find('-')
          if i > -1:
            I = np.hstack((I, np.arange(int(x[:i]), int(x[i+1:])+1, dtype = int)))
    else:
      I = np.hstack((I, int(x)))
  return np.unique(I).tolist()

