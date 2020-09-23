import numpy as np
import scipy as sp
import scipy.optimize as spo
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.dtypes import *

np.E = 0.57721566490153286060651209008240243104215933593992

def sortparam(_p, s, ofs, ply): # sorts parameter groups in order specified by s, where the p[:ofs] are ignored and p[ofs:] onwards are sorted into groups of size ply  
  n = len(s)
  m = len(_p) - ofs
  if m % ply:
    raise ValueError("Parameter dimensions incompatible with specified offset and ply-value")
  
  if m // ply != n:
    raise ValueError("Parameter dimensions incompataible with sort specification")
  
  p = _p[:ofs].copy()
  i = np.argsort(s)
  for j in range(n):
    k = ofs + ply * i[j]
    p = np.hstack([p, _p[(k):(k+ply)]])
  return p
  

def expdfun(p, x, opts = 0): # exponential decay function (either polarity, and variable offset)
  # opts = 0 returns value, 1 returns derivative, 2 returns both
  #y = p[0] + p[1] * exp (-x * exp(p[2])) + p[3] * exp (-x * exp(p[4])) + ....
  lp = len(p)
  lx = len(x)
 
  if lp == 3: # deal with single case first
    exponent = - x * expo(p[2])
    exponential = expo(exponent)
    expproduct = p[1] * exponential
    if opts != 2:
      y = p[0] + expproduct
    if not(opts): return y
    J = np.array( (np.ones(lx, dtype = float), exponential, expproduct * exponent) ) 
    if opts == 1: return J
    return y, J
  
  if (lp - 1) % 2:
    raise ValueError("Exponential decay parameter argument dimension of unknown specification")

  n = (lp - 1) // 2
  
  exponent = np.empty( (n, lx), dtype = float)
  exponential = np.empty( (n, lx), dtype = float)
  expproduct = np.empty( (n, lx), dtype = float)
  
  for i in range(n):
    exponent[i] =  - x * expo(p[2+i*2])
    exponential[i] = expo(exponent[i])
    expproduct[i] = p[1+i*2] * exponential[i]
  
  if opts != 2:
    y = p[0]
    for i in range(n):
      y += expproduct[i]
                     
  if not(opts): return y
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones( (lx), dtype = float)
  
  for i in range(n):
    J[1+i*2] = exponential[i]
    J[2+i*2] = expproduct[i] * exponent[i]
  
  if opts == 1: return J
  return y, J


def expdval(p, x, y = None): # returns residual if y is entered
  if y is None: return expdfun(p, x)
  return expdfun(p, x) - y 

def expdder(p, x, y = None):
  return expdfun(p, x, 1)

def expdfit(x, y, n = 1, p0 = None):
  lp = 2 * n + 1
  if len(x) < lp:
    raise ValueError("Insufficent data for exponential fit(s)")

  i, j = np.argmin(x), np.argmax(x)
  dy = y[i] - y[j]
  y0 = y.min() if dy >= 0. else y.max()
  if n > 1 and p0 is not None: # allow bypassing single-fits externally
    fit = [p0]
  else:
    estytau = y0 + dy / np.e
    esttau = x[np.argmin(np.fabs(y - estytau))] - x[i]
    estamp = dy if x[i] <= 0. else dy/np.exp(-x[i]/esttau)
      
    p0 = np.array([y0, estamp, -np.log(esttau)], dtype = float)
    fit = spo.leastsq(expdval, p0, args=(x,y), Dfun=expdder, col_deriv = 1)
    if np.isnan(fit[0][0]): # reset to initial estimates if isnan
      fit = list(fit); 
      fit[0] = p0
      fit = tuple(fit)
  
  if n == 1: return fit

  p = fit[0]
  p0 = np.empty(1+n*2, dtype = float)
  p0[0] = p[0]
  c = np.linspace(0.5, 1.5, n)
  d = -np.arange(n, dtype = float)
  d -= np.mean(d)

  k = 1
  for i in range(n):
    p0[k] = p[1] * c[i]
    p0[k+1] = p[2] + d[i]
    k += 2

  fit = list(spo.leastsq(expdval, p0, args=(x,y), Dfun=expdder, col_deriv = 1))
  fit[0] = sortparam(fit[0], -fit[0][2::2], 1, 2)
  return tuple(fit)

def exp0fun(p, x, opts = 0): # opts = 0 returns value, 1 returns derivative, 2 returns both
  #y = p[0] * exp (-x * exp(p[1])) + p[2] * exp (-x * exp(p[3])) + ....
  lp = len(p)
  lx = len(x)
 
  if lp == 2: # deal with single case first
    exponent = - x * expo(p[1])
    exponential = expo(exponent)
    expproduct = p[0] * exponential
    if not(opts): return expproduct
    J = np.array( (exponential, expproduct * exponent) ) 
    if opts == 1: return J
    return expproduct, J
  
  if lp % 2:
    raise ValueError("Exponential decay parameter argument dimension of unknown specification")

  n = lp // 2
  
  exponent = np.empty( (n, lx), dtype = float)
  exponential = np.empty( (n, lx), dtype = float)
  expproduct = np.empty( (n, lx), dtype = float)
  
  for i in range(n):
    exponent[i] =  - x * expo(p[1+i*2])
    exponential[i] = expo(exponent[i])
    expproduct[i] = p[i*2] * exponential[i]
  
  if opts != 2:
    y = expproduct[0]
    for i in range(1,n):
      y += expproduct[i]
                     
  if not(opts): return y
  
  J = np.empty( (lp, lx), dtype = float)
  
  for i in range(n):
    J[i*2] = exponential[i]
    J[i*2+1] = expproduct[i] * exponent[i]
  
  if opts == 1: return J
  return y, J

def exp0val(p, x, y = None): # returns residual if y is entered
  if y is None: return exp0fun(p, x)
  return exp0fun(p, x) - y 

def exp0der(p, x, y = None):
  return exp0fun(p, x, 1)

def exp0fit(x, y, n = 1):
  lp = 2 * n
  if len(x) < lp:
    raise ValueError("Insufficent data for exponential fit(s)")
  
  i, j = np.argmin(x), np.argmax(x)
  dy = y[i] - y[j]
  y0 = y.min() if dy >= 0. else y.max()
  estytau = y0 + dy / np.e
  esttau = x[np.argmin(np.fabs(y - estytau))] - x[i]
  estamp = dy if x[i] <= 0. else dy/np.exp(-x[i]/esttau)

  p0 = np.array([dy, -np.log(esttau)], dtype = float)
  fit = spo.leastsq(exp0val, p0, args=(x,y), Dfun=exp0der, col_deriv = 1)
  
  if n == 1: return fit

  p = fit[0]
  p0 = np.empty(n*2, dtype = float)
  d = -np.arange(n, dtype = float)
  d -= np.mean(d)
  k = 0
  for i in range(n):
    p0[k] = p[0] * float(i+1)
    p0[k+1] = p[1] * d[i]
    k += 2

  fit = list(spo.leastsq(exp0val, p0, args=(x,y), Dfun=exp0der, col_deriv = 1))
  fit[0] = sortparam(fit[0], -fit[0][1::2], 0, 2)
  return tuple(fit)

def expofun(_p, _x, opts = 0): # y is always positive
  #y = p[0] + exp(p[1]) * exp (-x * exp(p[2]) +  exp(p[3]) * ...
  lp, lx = len(_p), len(_x)
  if (lp - 1) % 2:
    raise ValueError("expofun: input parameter argument dimension of unknown specification")
  n = (lp - 1) // 2
  p, x = np.copy(_p), -_x
  p[1:] = expo(p[1:])

  ep = np.empty((n, lx), dtype = float)
  Ep = np.empty((n, lx), dtype = float)
  Pe = np.empty((n, lx), dtype = float)

  y = np.tile(p[0], x.shape)
  m = -x
  k = 1

  for i in range(n):
    ep[i] = x*p[k+1]
    Ep[i] = expo(ep[i])
    Pe[i] = p[k] * Ep[i]
    y += Pe[i]
    k += 2

  if not(opts): return y
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones(lx, dtype = float)
  
  k = 1
  for i in range(n):
    J[k] = Pe[i]
    J[k+1] = Pe[i]*ep[i]
    k += 2
  
  if opts == 1: return J

  return y, J

def expoval(p, x, y = None): # returns residual if y is entered
  if y is None: return expofun(p, x)
  return expofun(p, x) - y 

def expoder(p, x, y = None):
  return expofun(p, x, 1)

def expofit(_x, _y, n = 1):
  lp = 2 * n + 1
  if len(_x) < lp: raise ValueError("Insufficent data for exponential fit(s)")  

  i, j = np.argmin(x), np.argmax(x)
  dy = y[i] - y[j]
  y0 = y.min() if dy >= 0. else y.max()
  estytau = y0 + dy / np.e
  esttau = x[np.argmin(np.fabs(y - estytau))] - x[i]
  estamp = dy if x[i] <= 0. else dy/np.exp(-x[i]/esttau)
    
  p0 = np.array([y0, np.log(estamp), -np.log(esttau)], dtype = float)
  fit = spo.leastsq(expoval, p0, args=(x,y), Dfun=expoder, col_deriv = 1)

  if n == 1: return fit

  p0 = fit[0]
  p = np.empty(lp, dtype = float)
  p[0] = p0[0]
  k = 1
  for i in range(n):
    p[k:k+2] = p0[1:]
    c = float(2*i)
    p[k] += c
    p[k+1] -= c
    k += 2
  
  p0 = np.copy(p)
  fit = list(spo.leastsq(expoval, p0, args=(x,y), Dfun=expoder, col_deriv = 1))
  return tuple(fit) 

def exppulsefun(p, x, opts = 0):
  #y = p[0] + p[1] * exp ((p[2]-x) * exp(p[3]) * ( 1 - exp((p[2]-x) * exp(p[4])) + p[5] * ...
  if type(p) is list: p = np.array(p, dtype = float)
  lp, lx = len(p), len(x)
  
  if (lp - 1) % 4:
    raise ValueError("Exponential pulse parameter argument dimension of unknown specification") 
  
  n = (lp-1)//4;  
  if n == 1:
    mult = np.array(p[1]).reshape(1)
    xoff = np.array(p[2]).reshape(1)
    posc = np.array(expo(p[3])).reshape(1)
    negc = np.array(expo(p[4])).reshape(1)
  else:
    i = 4*np.arange(n, dtype = int) + 1
    mult = p[i]; i += 1
    xoff = p[i]; i += 1
    posc = expo(p[i]); i += 1
    negc = expo(p[i]);
  
  negx = np.empty((n, lx), dtype = float)
  expp = np.empty((n, lx), dtype = float)
  expn = np.empty((n, lx), dtype = float)
  expd = np.empty((n, lx), dtype = float)
  expm = np.empty((n, lx), dtype = float)
  
  for i in range(n):
    negx[i] = xoff[i] - x
    expp[i] = expo(negx[i] * posc[i])
    expn[i] = expo(negx[i] * (posc[i]+negc[i]))
    expd[i] = expp[i] - expn[i]
    expm[i] = mult[i] * expd[i]
    
  if opts != 2:
    y = p[0]
    for i in range(n):    
      y += expm[i]
                      
  if not(opts):
    return y    
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones((lx), dtype = float)
  
  for i in range(n):
    sumc = posc[i] + negc[i]
    J[1+i*4] = expd[i]
    J[2+i*4] = mult[i] * expp[i] * (posc[i]*expp[i] - sumc*expn[i])
    J[3+i*4] = negx[i] * posc[i] * expm[i]
    J[4+i*4] = - mult[i] * negx[i] * negc[i] * expn[i]
  
  if opts == 1:
    return J
  return y, J

def exppulsefun(p, x, opts = 0):
  #y = p[0] + p[1] * exp ((p[2]-x) * exp(p[3]) * ( 1 - exp((p[2]-x) * exp(p[3])) + p[4] + ...
  
  lp = len(p)
  lx = len(x)
  
  if (lp - 1) % 3:
    raise ValueError("Exponential decay parameter argument dimension of unknown specification")

  n = lp // 3

  mult = np.empty( (n), dtype = float)
  toff = np.empty( (n), dtype = float)
  tdec = np.empty( (n), dtype = float)

  for i in range(n):
    mult[i] = p[1+i*3]
    toff[i] = p[2+i*3]
    tdec[i] = p[3+i*3]
    
  decc = expo(tdec)
  offc = toff * decc
  
  expminx = np.empty( (n, lx), dtype = float) 
  expnent = np.empty( (n, lx), dtype = float) 
  exptial = np.empty( (n, lx), dtype = float)
  expprod = np.empty( (n, lx), dtype = float)
    
  for i in range(n):
    expminx[i] = -decc[i] * x
    expnent[i] = offc[i] + expminx[i]
    exptial[i] = expo(expnent[i])
    expprod[i] = exptial[i] * (1.0 - exptial[i])
    
  if opts != 2:
    y = p[0]
    for i in range(n):
      y += mult[i] * expprod[i]
                     
  if not(opts):
    return y
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones( (lx), dtype = float)
  
  for i in range(n):
    derprod = mult[i] * exptial[i] * (1.0 - 2.0 * exptial[i])
    J[1+i*3] = expprod[i]
    J[2+i*3] = derprod * decc[i]
    J[3+i*3] = derprod * expminx[i]
  
  if opts == 1:
    return J
  return y, J

def exppulseval(p, x, y = None): # returns residual if y is entered
  if y is None: return exppulsefun(p, x)
  return exppulsefun(p, x) - y 

def exppulseder(p, x, y = None):
  return exppulsefun(p, x, 1)
  
def exppulsefit(x, y, n = 1, siglogamp = None):
  if siglogamp == None:
    siglogamp = 10 # np.e ** (np.e ** np.E) 
  
  mino = 1e-300
  lp = 3 * n + 1
    
  if len(x) < lp:
    raise ValueError("Insufficent data for exponential fit(s)")  
    
  p0 = np.empty(4, dtype = float)
  
  ilt = np.argmin(x)
  irt = np.argmax(x)
  xlt = x[ilt]
  xrt = x[irt]
  yrt = y[irt]
  xrn = x[irt] - x[ilt]
  
  p0[0] = yrt
  
  imx = np.argmax(y)
  imn = np.argmin(y)
    
  ymx = y[imx]
  ymn = y[imn]
  
  mxi = max(imx, imn)
  mni = min(imx, imn)  
  
  if (mxi - mni) < (irt - mxi):
    ind = np.nonzero(x > x[mxi])
  else:
    ind = np.nonzero(x > x[mni])
    
  xind = x[ind]
  yind = y[ind]
  
  cc = np.corrcoef(xind, yind)  
  isd = cc[0][1] < 0        
  ran = ymx - yrt if isd else yrt - ymn
  
  if isd:
    p0[1] = ran * 4.0
    ilo = imx
    estvaltau = y[ilo] - ran*(1.0 - 1.0/np.e)
    
  else: 
    p0[1] = -ran * 4.0
    ilo = imn
    estvaltau = y[ilo] + ran*(1.0 - 1.0/np.e) 
    
  p0[2] = xlt
  
  ind = np.nonzero(x > x[ilo])
  xind = x[ind]
  yind = y[ind]  
  htau = xind[np.argmin(np.fabs(yind-estvaltau))] - x[ilo] + mino
  
  p0[3] = -np.log(htau)
  
  #pl.plot(x, y, 'b')
  #pl.hold(True)
  
  fit = spo.leastsq(exppulseval, p0, args=(x,y), Dfun=expulsepder, col_deriv = 1)
  
  #pl.plot(x, exppval(fit[0], x), 'r')

  if n == 1:
    return fit
  
  if fit[0][2] >= xlt-xrn and fit[0][2] <= xrt+xrn:
    p0 = fit[0][0]  
    p = fit[0][1:]  
  else:
    p = p0[1:] + 0.0
    p0 = p0[0]
    
  p[1] = np.maximum(xlt, p[1])
  
  for i in range(n):
    p0 = np.hstack([p0, p[:]])  
    
  ldci = np.arange(3, lp, 3)  
  m = np.arange(n, dtype = float)
  p0[ldci] += m * np.e
    
  fit = list(spo.leastsq(exppulseval, p0, args=(x,y), Dfun=exppulseder, col_deriv = 1))
  fit[0] = sortparam(fit[0], -exppinf(fit[0]), 1, 3)  

  ampi = np.arange(1, lp, 3)
  lamp = np.log(np.fabs(fit[0][ampi]))
  maxi = lamp.argmax()
  loai = np.nonzero(lamp < (lamp[maxi] - siglogamp)); loai = loai[0]
  if len(loai):    
    i0mx = int(maxi * 3 + 1)
    i0lo = int(loai * 3 + 1)
    snmx = float(np.sign(fit[0][i0mx]))
    for i in range(len(loai)):
      i0 = i0lo[i]
      fit[0][i0] = np.fabs(fit[0][i0]) * snmx
      fit[0][i0+1] = fit[0][i0mx+1]
      fit[0][i0+2] = fit[0][i0mx+2]
    fit[0] = sortparam(fit[0], -np.fabs(fit[0][ampi]), 1, 3)   
    
  #pl.plot(x, y)
  #pl.hold(True)
  #pl.plot(x, exppval(fit[0], x), 'r')
  #print(fit[0])
  #pl.show()
  
  return tuple(fit) 
 
def exppulseinf(p): # returns x values of inflection point for each component
  mino = 1e-300
  n = len(p) // 3  
  x = np.empty( (n), dtype = float)  
  for i in range(n):    
    x[i] = p[2+i*3] - np.log(0.5)/(mino+expo(p[3+i*3]))        
  return x  


def exppulsetau(p): # returns time constant for each component
  n = len(p) / 3
  t = np.empty( (n), dtype = float)
  for i in range(n):
    t[i] = expo(-p[3+i*3])  
  return t  


def expmfun(p, x, opts = 0):
  #y = p[0] - exp (p[1] + (p[2]-x) * exp(p[3]) * ( 1 - exp((p[2]-x) * exp(p[3])) - exp(p[4] + ...
  
  lp = len(p)
  lx = len(x)
  
  if (lp - 1) % 3:
    raise ValueError("Exponential decay parameter argument dimension of unknown specification")

  n = lp // 3

  tamp = np.empty( (n), dtype = float)
  toff = np.empty( (n), dtype = float)
  tdec = np.empty( (n), dtype = float)

  for i in range(n):
    tamp[i] = p[1+i*3]
    toff[i] = p[2+i*3]
    tdec[i] = p[3+i*3]
    
  ampc = -expo(tamp)  
  decc = expo(tdec)
  offc = toff * decc
  
  expminx = np.empty( (n, lx), dtype = float) 
  expnent = np.empty( (n, lx), dtype = float) 
  exptial = np.empty( (n, lx), dtype = float)
  expprod = np.empty( (n, lx), dtype = float)
    
  for i in range(n):
    expminx[i] = -decc[i] * x
    expnent[i] = offc[i] + expminx[i]
    exptial[i] = expo(expnent[i])
    expprod[i] = ampc[i] * exptial[i] * (1.0 - exptial[i])
    
  if opts != 2:
    y = p[0]
    for i in range(n):
      y += expprod[i]
                     
  if not(opts):
    return y
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones( (lx), dtype = float)
  
  for i in range(n):
    derprod = ampc[i] * exptial[i] * (1.0 - 2.0 * exptial[i])
    J[1+i*3] = expprod[i]
    J[2+i*3] = derprod * decc[i]
    J[3+i*3] = derprod * expminx[i]
  
  if opts == 1:
    return J
  return y, J

def expmval(p, x, y = None): # returns residual if y is entered
  if y is None: return expmfun(p, x)
  return expmfun(p, x) - y 

def expmder(p, x, y = None):
  return expmfun(p, x, 1)
  
def expmfit(_x, _y, n = 1):
  lp = 3 * n + 1
    
  if len(_x) < lp:
    raise ValueError("Insufficent data for exponential fit(s)")  

  i = np.argsort(_x)
  x, y = _x[i], _y[i]
  yo, ye = y[0], y[-1]
  yr = y.max() - y.min()
    
  p0 = np.empty(4, dtype = float)
  p0[0] = y[-1]
  p0[1] = np.log(yr)
  p0[2] = x[0]
  p0[3] = -np.log(x[np.nonzero(y < yo - 0.6*yr)[0][0]] - x[0])
  
  fit = spo.leastsq(expmval, p0, args=(x,y), Dfun=expmder, col_deriv = 1)

  if n == 1: return fit

  p0 = fit[0]
  p0[2] = max(p0[2], x[0])
  p = np.empty(lp, dtype = float)
  d = -np.arange(n, dtype = float)
  d -= np.mean(d)
  p[0] = p0[0]
  k = 1
  for i in range(n):
    p[k:k+3] = p0[1:]
    p[k] += float(i)
    #p[k+1] *= float(i+1.)
    p[k+2] += d[i]
    k += 3
  
  p0 = np.copy(p)
  fit = list(spo.leastsq(expmval, p0, args=(x,y), Dfun=expmder, col_deriv = 1))
  fit[0] = sortparam(fit[0], -exppinf(fit[0]), 1, 3)  
  #fit[0] = p0
  '''

  ampi = np.arange(1, lp, 3)
  lamp = fit[0][ampi]
  maxi = lamp.argmax()
  loai = np.nonzero(lamp < (lamp[maxi] - siglogamp)); loai = loai[0]
  if len(loai):    
    i0mx = maxi * 3 + 1    
    i0lo = loai * 3 + 1
    for i in range(len(loai)):
      i0 = i0lo[i]
      fit[0][i0] = fit[0][i0mx] - siglogamp
      fit[0][i0+1] = fit[0][i0mx+1]
      fit[0][i0+2] = fit[0][i0mx+2]
    fit[0] = sortparam(fit[0], -fit[0][ampi], 1, 3)   
  ''' 
  return tuple(fit) 

def exppfun(_p, _x, opts = 0):
  #y = p[0] + exp(p[1]) * exp (-x * exp(p[2]) -  exp(p[3]) * exp(-x * exp(p[4])) + exp(p[5]) * ...
  lp, lx = len(_p), len(_x)
  if (lp - 1) % 4:
    raise ValueError("exppfun: input parameter argument dimension of unknown specification")
  n = (lp - 1) // 4
  p, x = np.copy(_p), -_x
  p[1:] = expo(p[1:])

  ep = np.empty((n, lx), dtype = float)
  en = np.empty((n, lx), dtype = float)
  Ep = np.empty((n, lx), dtype = float)
  En = np.empty((n, lx), dtype = float)
  Pe = np.empty((n, lx), dtype = float)
  Ne = np.empty((n, lx), dtype = float)

  y = np.tile(p[0], x.shape)
  m = -x
  k = 1

  for i in range(n):
    ep[i] = x*p[k+1]
    en[i] = x*p[k+3]
    Ep[i] = expo(ep[i])
    En[i] = expo(en[i])
    Pe[i] = p[k] * Ep[i]
    Ne[i] = -p[k+2] * En[i]
    y += Pe[i] + Ne[i]
    k += 4

  if not(opts): return y
  
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones(lx, dtype = float)
  
  k = 1
  for i in range(n):
    J[k] = Pe[i]
    J[k+1] = Pe[i]*ep[i]
    J[k+2] = Ne[i]
    J[k+3] = Ne[i]*en[i]
    k += 4
  
  if opts == 1: return J

  return y, J

def exppval(p, x, y = None): # returns residual if y is entered
  if y is None: return exppfun(p, x)
  return exppfun(p, x) - y 

def exppder(p, x, y = None):
  return exppfun(p, x, 1)

def exppinf(p):
  return (p[1]+p[2] - p[3]-p[4]) / (expo(p[2]) - expo(p[4]))
  
def exppamp(p):
  return exppval(p, exppinf(p)) - p[0]
  
def exppfit(_x, _y, n = 1):
  lp = int(4 * n + 1)
  if len(_x) < lp: raise ValueError("Insufficent data for exponential fit(s)")  

  _i = np.argsort(_x)
  x, y = _x[_i], _y[_i]
  yo, ye = y[0], y[-1]
  imax, imin = y.argmax(), y.argmin()
  i = min(int(len(x)/2.), max(imax, imin))
  cc = np.corrcoef(x[i:], y[i:])  
  pol = 1. if cc[0][1] < 0 else -1.
  if pol > 0.:
    ofs = y[imin]
    ind = imax
    amp = y[imax] - y[imin]
  else:
    ofs = y[imax]
    ind = imin
    amp = y[imin] - y[imax]
  ons = x[1]
  tau = x[ind] - x[0] if ind else x[-1] - x[0]
  lam = np.log(np.fabs(amp))
  ldc = -np.log(tau*0.63212) # 0.63212 = (1.-1./np.e)

  p0 = np.empty(5, dtype = float)
  p0[0] = ofs
  p0[1] = lam - pol
  p0[2] = ldc - pol
  p0[3] = lam + pol
  p0[4] = ldc + pol
  
  fit = spo.leastsq(exppval, p0, args=(x,y), Dfun=exppder, col_deriv = 1)
  #print(p0, fit[0])
  #print(exppinf(fit[0]), x[np.argmin(y)])

  if n == 1: return fit

  p0 = fit[0]
  p = np.empty(lp, dtype = float)
  p[0] = p0[0]
  k = 1
  for i in range(n):
    p[k:k+4] = p0[1:]
    c = float(2*i)
    p[k] += c
    p[k+1] -= c
    p[k+2] += c
    p[k+3] -= c
    k += 4
  
  p0 = np.copy(p)
  fit = list(spo.leastsq(exppval, p0, args=(x,y), Dfun=exppder, col_deriv = 1))
  return tuple(fit) 

def exp2fun(_p, _x, opts = 0):
  #y = p[0] + exp(p[1]) * (exp(-x * exp(p[2])) - exp(-x * exp([3]))) + exp([4]) *  ...
  # - note this functional form accomodates both polarities
  lp, lx = len(_p), len(_x)
  if (lp - 1) % 3:
    raise ValueError("exppfun: input parameter argument dimension of unknown specification")
  n = (lp - 1) // 3
  p, x = np.copy(_p), -_x
  p[1:] = expo(p[1:])

  ep = np.empty((n, lx), dtype = float)
  en = np.empty((n, lx), dtype = float)
  Ep = np.empty((n, lx), dtype = float)
  En = np.empty((n, lx), dtype = float)
  de = np.empty((n, lx), dtype = float)
  De = np.empty((n, lx), dtype = float)

  y = np.tile(p[0], x.shape)
  k = 1

  for i in range(n):
    ep[i] = x*p[k+1]
    en[i] = x*p[k+2]
    Ep[i] = expo(ep[i])
    En[i] = expo(en[i])
    De[i] = p[k] * (Ep[i] - En[i])
    y += De[i]
    k += 3

  if not(opts): return y
  
  '''
  J = np.empty( (lp, lx), dtype = float)
  J[0] = np.ones(lx, dtype = float)
  '''
  J = np.ones( (lp, lx), dtype = float)
  
  k = 1
  for i in range(n):
    J[k] = De[i]
    J[k+1] =  p[k] * ep[i] * Ep[i]
    J[k+2] = -p[k] * en[i] * En[i]
    k += 3
  
  if opts == 1: return J

  return y, J

def exp2val(p, x, y = None): # returns residual if y is entered
  if y is None: return exp2fun(p, x)
  return exp2fun(p, x) - y 

def exp2der(p, x, y = None):
  return exp2fun(p, x, 1)

def exp2inf(p):
  lp = len(p)
  if (lp - 1) % 3:
    raise ValueError("exppfun: input parameter argument dimension of unknown specification")
  n = (lp - 1) // 3
  x = np.empty(n, dtype = float)
  k = 1
  for i in range(n):
    x[i] = (p[k+1] - p[k+2]) / (expo(p[k+1]) - expo(p[k+2]))
    k += 3
  return x
  
def exp2amp(p):
  return exp2val(p, exp2inf(p)) - p[0]
  
def exp2fit(_x, _y, n = 1, pol = None):
  lp = 3 * n + 1
  if len(_x) < lp: raise ValueError("Insufficent data for exponential fit(s)")  
  _i = np.argsort(_x)
  x, y = _x[_i], _y[_i]
  Tau = x[-1] - x[0]
  Tau = [Tau/float(len(x)), Tau]
  imax, imin = y.argmax(), y.argmin()
  if pol is None:
    pol = -1. if isconvex(y, 3) else 1.
  if pol > 0.:
    ofs = y[imin]
    tau = x[imax]
  else: 
    ofs = y[imax]
    tau = x[imin]
  amp = y[imax] - y[imin]
  lam = np.log(np.fabs(amp))
  tau = max(Tau[0], min(Tau[1], tau))
  ldc = -np.log(tau)

  p0 = np.empty(4, dtype = float)
  p0[0] = ofs
  p0[1] = lam + 1.
  if pol > 0:
    p0[2] = ldc - 0.5
    p0[3] = ldc + 0.5
  else:
    p0[2] = ldc + 0.5
    p0[3] = ldc - 0.5


  fit = spo.leastsq(exp2val, p0, args=(x,y), Dfun=exp2der, col_deriv = 1)
  if np.isnan(fit[0][0]): # reset to initial estimates if isnan
    fit = list(fit); 
    fit[0] = p0
    fit = tuple(fit)
  #print(p0, fit[0])
  #print(exppinf(fit[0]), x[np.argmin(y)])

  if n == 1: return fit

  p0 = fit[0]
  p = np.empty(lp, dtype = float)
  p[0] = p0[0]
  k = 1
  d = np.arange(n-1, -1, -1, dtype = float)
  for i in range(n):
    p[k:k+3] = p0[1:]
    p[k+1] += d[i]
    p[k+2] += d[i]
    k += 3
  
  p0 = np.copy(p)
  #fit = list(spo.leastsq(exp2val, p0, args=(x,y)))
  fit = list(spo.leastsq(exp2val, p0, args=(x,y), Dfun=exp2der, col_deriv = 1))
  fit[0] = sortparam(fit[0], exp2inf(fit[0]), 1, 3)
  #print(fit[0])
  return tuple(fit) 

def htanfun(p, x, opts = 0): # opts = 0 returns value, 1 returns derivative, 2 returns both
  # returns p[0] + p[1] * tanh ( (x - p[2]) * exp(p[3]))
  expt = expo(p[3])
  htnt = (x - p[2]) * expt
  htan = np.tanh(htnt)
  y = p[0] + p[1] * htan
  if not(opts): return y
  J = np.empty( (4, len(x)), dtype = float)
  dert = p[1] * (1 - htan ** 2)
  J[0] = np.ones(len(x))
  J[1] = htan
  J[2] = dert * -expt
  J[3] = dert * htnt
  if opts == 1: return J
  return y, J
  
def htanval(p, x, y = None): # returns residual if y is entered
  if y is None: return htanfun(p, x)
  return htanfun(p, x) - y 

def htander(p, x, y = None):
  return htanfun(p, x, 1)  

def htanfit(x, y):
  p0 = np.zeros(4, dtype = float)
  ilt = np.argmin(x)
  irt = np.argmax(x)
  ymax = y.max()
  ymin = y.min()
  ylt = y[ilt]
  yrt = y[irt]
  ran = ymax - ymin if ylt <= yrt else ymin - ymax 
  p0[0] = np.mean(y)  
  p0[1] = ran / 2.0
  p0[2] = np.mean(x)
  p0[3] = -np.log(x[irt] - x[ilt])
  fit = spo.leastsq(htanval, p0, args=(x,y), Dfun=htander, col_deriv = 1)
  return fit
  
def sigmfun(p, x, opts = 0):
  # returns p[0] + p[1] / (1 + exp( (p[2] - x)*exp(p[3])))
  expt = expo(p[3])
  expn = (p[2] - x) * expt
  expe = expo(expn)
  divi = 1. + expo(expe)
  quot = p[1] / divi
  y = p[0] + quot
  if not(opts): return y
  mqds = -quot * expe / divi
  J = np.empty( (4, len(x)), dtype = float)
  J[0] = np.ones(len(x))
  J[1] = 1. / divi
  J[2] = mqds * expt
  J[3] = mqds * expn
  if opts == 1: return J
  return y, J

def sigmval(p, x, y = None): # returns residual if y is entered
  if y is None: return sigmfun(p, x)
  return sigmfun(p, x) - y 

def sigmder(p, x, y = None):
  return sigmfun(p, x, 1)  

def sigmfit(x, y):
  ilt = np.argmin(x)
  irt = np.argmax(x)
  ylt = y[ilt]
  yrt = y[irt]
  ymax = y.max()
  ymin = y.min()
  ran = ymax - ymin if ylt <= yrt else ymin - ymax 
  p0 = np.empty(4, dtype = float)
  p0[0] = ylt
  p0[1] = ran*1.1
  p0[2] = np.mean(x)
  p0[3] = 1.6487-np.log(x[irt] - x[ilt])
  fit = spo.leastsq(sigmval, p0, args=(x,y), Dfun=sigmder, col_deriv = 1)
  return fit

def sig0fun(p, x, opts = 0):
  # returns p[0] / (1 + exp( (p[1] - x)*exp(p[2])))
  expt = expo(p[2])
  expn = (p[1] - x) * expt
  expe = expo(expn)
  divi = 1. + expo(expe)
  quot = p[0] / divi
  if not(opts): return quot
  mqds = -quot * expe / divi
  J = np.empty( (3, len(x)), dtype = float)
  J[0] = 1. / divi
  J[1] = mqds * expt
  J[2] = mqds * expn
  if opts == 1: return J
  return J
  #return quot, J

def sig0val(p, x, y = None): # returns residual if y is entered
  if y is None: 
    return sig0fun(p, x)
  return sig0fun(p, x) - y 

def sig0der(p, x, y = None):
  return sig0fun(p, x, 1)  

def sig0fit(x, y):
  ilt = np.argmin(x)
  irt = np.argmax(x)
  ylt = y[ilt]
  yrt = y[irt]
  ymax = y.max()
  ymin = y.min()
  ran = ymax if ylt <= yrt else ymin
  p0 = np.empty(3, dtype = float)
  p0[0] = ran*1.2
  p0[1] = np.mean(x)
  p0[2] = 1.6487-np.log(x[irt] - x[ilt])
  fit = spo.leastsq(sig0val, p0, args=(x,y), Dfun=sig0der, col_deriv = 1)
  #print(p0, fit[0])
  return fit

def idftfun(p, x, opts = 0): # opts = 0 returns value, 1 returns derivative, 2 returns both
  # returns 0.5p[0] + p[1]cos(x) + p[2]sin(x) + p[3]cos(2x) + p[4]sin(2x) + ...
  c0 = 1.
  creal = True
  ssign = -1.
  lp, lx = len(p), len(x)
  if not(lp % 2): raise ValueError("Parameter vector must be odd in length")
  n = lp // 2
  c = np.empty((n, lx), dtype = float)
  s = np.empty((n, lx), dtype = float)
  J = np.tile(c0, (lp, lx))
  y = np.tile(c0*p[0], lx)
  for i in range(n):
    ix  = float(i+1) * x
    c[i] = np.cos(ix)
    s[i] = ssign * np.sin(ix)
    if creal:
      J[2*i+1] = c[i]
      J[2*i+2] = s[i]
      y += p[2*i+1] * c[i] + p[2*i+2] * s[i]
    else:
      J[2*i+1] = s[i]
      J[2*i+2] = c[i]
      y += p[2*i+1] * s[i] + p[2*i+2] * c[i]
  if opts == 0: return y
  if opts == 1: return J
  return y, J

def idftval(p, x, y = None): # returns residual if y is entered
  if y is None: return idftfun(p, x)
  return idftfun(p, x) - y 

def idftder(p, x, y = None):
  return idftfun(p, x, 1)  

def idftfit(x, y, n = None):
  nx = len(x)
  ny = len(y)
  if nx != ny: raise ValueError("x/y inputs incommensurate.")
  if n is None: n = nx // 2
  p0 = np.zeros(n*2+1, dtype = float)
  p0[0] = .5 * np.mean(y)
  fit = spo.leastsq(idftval, p0, args=(x,y), Dfun=idftder, col_deriv = 1)
  return fit

