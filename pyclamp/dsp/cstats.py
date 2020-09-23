# A module for circular statistics

import numpy as np
import scipy.stats as stats

def cang(X, maxx = np.pi):           # -pi <= angle <= pi
  twopi = 2.*np.pi
  x = np.mod(X, twopi) 
  if maxx >= twopi:
    return x
  if type(X) is np.ndarray:
    if x.ndim:
      x[x>maxx] -= twopi
    elif x > maxx:
      x -= twopi
  elif x > maxx:
    x -= twopi
  return x  

def crad(X, ran, maxx = np.pi):
  return cang(2.*np.pi*(X-np.mean(ran))/np.diff(ran), maxx)

def cmean(X, **kwargs): # -pi <= circular mean <= +pi
  x = cang(X)
  c = np.sum(np.cos(x), **kwargs)
  s = np.sum(np.sin(x), **kwargs)
  return np.arctan2(s, c)

def cmrl(X, **kwargs): # 0 <= mean resultant length <= 1
  x = cang(X)
  c = np.sum(np.cos(x), **kwargs)
  s = np.sum(np.sin(x), **kwargs)
  r = np.sqrt(c**2. + s**2.)
  n = float(len(np.ravel(X))) / float(len(np.ravel(r)))
  return r/n

def cvar(X, **kwargs): # 0 <= circular variance <= 1
  return 1. - cmrl(X, **kwargs)

def cdev(X, **kwargs): # 0 <= circular deviation <= sqrt(2)
  return np.sqrt(2. * cvar(X, **kwargs))

def cdis(X, **kwargs): # 0 <= circular dispersion <= inf
  return -2. * np.log(cmrl(X, **kwargs))

def cstd(X, **kwargs): # 0 <= circular st.dev. <= inf
  return np.sqrt(cdis(X, **kwargs))

def ccon(X, opts = 0, kn = 15): # 0 <= circular concentration <= inf
  x = np.ravel(X)
  # opts = 0 vaguely accurate for large k
  if opts != 1 and opts != 2: return 1. / cdis(x)
  # opts = 1 based on Bessel approximation
  '''
  from p.88 of
  @book{fisher1995statistical,
  title={Statistical analysis of circular data},
  author={Fisher, Nicholas I},
  year={1995},
  publisher={Cambridge University Press}
  }
  '''
  n = len(x)
  r = cmrl(x)
  if r < 0.53:
    k = 2.*r + r**3. + 5.*r**5./6.
  elif r < 0.85:
    k = -0.4 + 1.39*r + 0.43/(1.-r)
  else:
    k = 1./(r**3. - 4.*r**2. + 3.*r)
  if opts == 1: return k 
  # opts = 2 reduces bias for low n
  if n <= kn:
    n = float(n)
    if k < 2:
      k = max(k - 2./(n*k), 0.)
    else:
      k = (n-1.)**3. * k / (n**3. + n)
  return k    

def vmpdf(X, M = 0., k = 1., kn = 230.): # von mises pdf with normal approximation from k >= kn
  m = cang(M)
  x = cang(X, m+np.pi)
  if k < kn:
    return stats.vonmises.pdf(x-m, k)
  # Normal approximation
  return stats.norm.pdf(x-m, loc = 0., scale = 1./np.sqrt(k))

