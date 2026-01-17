# A time-point module

import numpy as np
import scipy as sp
from pyclamp.dsp.sifunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.dtypes import *
from pyclamp.dsp.optfunc import *

MAXSI32 = 9223372036854775807

def latency(_t, _r, d = 0., n = 0, o = False): 
  # calculates latency of t wrt reference r offset d assuming overlap o
  # if n != 0, then the nth latencies are returned
  # if n < 0, then the interval preceding (if present) is returned
  t, r, absn = nanravel(_t) + d, nanravel(_r), abs(n)

  if not(o): # deal with the easy case of non-overlapping windows
    i = arghist(t, r)
    i[i < 0] = 0
    l = t - r[i] - d
    if n == 0: return l, i
    imax = i.max()
    _l = []
    _i = []
    for k in range(imax):
      ind = np.nonzero(i == k)[0]
      lind = l[ind]
      latk = lind[lind >= 0.] if n >= 0 else lind[lind < 0]
      if len(latk) >= absn:
        latk = np.sort(np.fabs(latk))
        lk = latk[absn-1] if n >= 0 else -latk[absn-1]
        _l.append(lk)
        _i.append(k)
    return np.array(_l), np.array(_i)
  nt, nr = len(t), len(r)
  mint, maxt = np.min(t), np.max(t)
  minr, maxr = np.min(r), np.max(r)
  l = np.empty(nt, dtype = float)
  i = np.empty(nt, dtype = int)

def lat2time(_t, t0 = 0., DT = 0.): # converts by-episode sorted latencies to absolute event and ref times
  #  _t = latency times
  # _t0 = reference time
  #  DT = summated sweep time
  t = np.ravel(_t)
  if not(len(t)): return _t, t0
  if isfloat(t0): t0 = np.tile(t0, 1)
  dt = np.hstack((0., np.diff(t)))
  if DT == 0:
    tmin = min(np.min(t0), np.min(t))
    tmax = max(np.max(t0), np.max(t))
    t0mx = 0. if len(t0) <= 1 else np.max(np.diff(t0))
    DT = max(np.max(dt), max(tmax - tmin, t0mx))
  nb = dt < 0.
  ne = len(np.nonzero(nb)[0]) + 1
  if ne > 1 and len(t0) == 1: t0 = np.tile(t0, ne)
  if len(t0) < ne: print("Warning from lat2time(): ambiguous reference time assignment")
  DT0 = np.cumsum(np.array(nb, dtype = float)) * DT
  return t + DT0, t0 + np.arange(len(t0), dtype = float) * DT

def interv(_t, _r, d = 0): # returns nearest intervening intervals offseting interval index by d
  t, r = np.sort(_t), np.sort(_r)
  nt, nr = len(t), len(r)
  k = arghist(r, t) + d
  h = np.tile(np.nan, nr)
  for i in range(nr):
    j = k[i]
    if j > 0 and j < nt-1:
      h[i] = t[j+1] - t[j]
  return np.array(h, dtype = float)

def latint(l, i = None, w = [-np.inf, 0.]): # collates interval within window for latencies l with indices i
  if i is None: i = np.cumsum(np.array(np.hstack((False, np.diff(l) < 0)), dtype = int))
  if len(l) != len(i): raise ValueError("Latency and index arrays non-commensurate.")
  D = np.array([], dtype = float)
  u = np.unique(i)
  n = len(u)
  for k in range(n):
    j = np.nonzero(i == u[k])[0]
    if len(j) > 1:
      t = l[j]
      j = np.nonzero(np.logical_and(t >= w[0], t < w[1]))[0]
      if len(j) > 1:
        D = np.hstack((D, np.diff(t[j])))
  return D

def excerpt(x, _t, _si = 1., window = None):
  if type(x) is not np.ndarray: x = np.array(x)
  if isint(_t): _t = np.array(_t)
  if window is None: window = [-np.inf, np.inf]
  window = np.array(window)
  t, si = np.sort(_t), float(_si)
  N = len(x)
  n = len(t)
  if not(n): return np.array([np.array([], x.dtype)])
  winf = np.isinf(window)
  if np.any(winf):
    hmindifft = 0.5 * min(np.diff(t))
    if winf[0]: window[0] = -min(t[0], hmindifft)
    if winf[1]: window[1] =  min(float(N)*si - t[-1], hmindifft) 
  i0 = np.array((t + window[0]) / si, dtype = int)
  i1 = np.array((t + window[1]) / si, dtype = int)
  i = np.nonzero(np.logical_and(np.logical_and(i0 >= 0, i0 < N),
                                np.logical_and(i1 >= 1, i1 < N)))[0]
  i0, i1 = i0[i], i1[i]
  n = len(i)
  if not(n): return np.array([np.array([], x.dtype)])
  i1 = i0 + np.min(i1 -i0)
  X = [[]] * n
  for i in range(n):
    X[i] = x[i0[i]:i1[i]]
  return np.array(X) 

def cusum(_x, normconst = 1, x0 = 0.):
  x = np.sort(_x)
  n = len(x)
  obsc = np.arange(n)
  xc = x[x < x0]
  nc = len(xc)
  if nc < 2: raise ValueError("Offset argument incompatible with data.")
  ranxc = xc[-1] - xc[0]
  if ranxc <= 0.: raise ValueError("Offset range incompatible with data.")
  expc = float(nc-1) * (x - xc[0]) / (ranxc)
  return x, (obsc - expc)/float(normconst)
  
def cusum1st(_x, _y, _x0 = 0, _nran = [4, np.inf], _xran = [-np.inf, np.inf], _pol = None): # returns polarity and end-time of first response
  i = np.argsort(_x)
  nran, xran = np.sort(_nran), np.sort(_xran)

  # Polarity decision first: joint vote between index and amplitude THEN overall change

  x_, y_, x0 = _x[i], _y[i], float(_x0)
  i = np.nonzero(np.logical_and(x_ >= x0, x_ <= xran[1]))[0]
  n = min(len(i), nran[1])
  if n < nran[0]:
    print("Warning: cusum1st() called with insufficient data")
    return None, None
  x, y = x_[i], y_[i]
  y -= y[0]
  il, ih = np.argmin(y), np.argmax(y)
  lo, hi = y[il], y[ih]

  indinh = il < ih
  ampinh = np.fabs(lo) > np.fabs(hi)

  if _pol is None:
    if indinh == ampinh:
      isinh = indinh
    else:
      isinh = y[0] > y[-1]
  else:
    pol = _pol
    isinh = pol < 0

  # determine limits of first response after rescaling cusum to global mean

  rany = y_[-1] - y_[0]
  if np.fabs(rany) >= 1./float(n):
    y_ -= np.linspace(0., rany, len(y_))
    y = y_[i]
  p0 = 1./float(n)

  done = False
  fail = 0

  while not(done): # we will switch polarity if necessary
    ri = np.inf if isinh else -np.inf
    r = np.tile(ri, n)
    for i in range(n):
      pi = np.nan
      if i >= nran[0] and x[i] >= xran[0]: ri, pi = sp.stats.spearmanr(x[:i], y[:i])
      if not(np.isnan(pi)):
        if pi <= p0:
          r[i] = ri * np.log(float(i)) # weight scores according to log length
          done = True
    if not(done):
      if fail:
        done = True
        fail += 1
      else:
        fail += 1
        if _pol is None: isinh = not(isinh)
  if isinh:
    pol = -1.
    i = np.argmin(r)
  else:
    pol = +1.
    i = np.argmax(r)
  xi = x[i]      
  if fail == 2:
    print("Warning from cusum1st(): no ascent or descent detected")
    pol = _pol
    xi = max(xran[0], x[nran[0]])

  return pol, xi;

def cusum1sthtanfit(lt, ne = 1, nran = [4, np.inf], xran = [-np.inf, np.inf], fran = [-0., +1.0], _pol = None): # returns x, y, (pol, e), fit[0]
  x, y, = cusum(lt, ne)
  pol, e = cusum1st(x, y, 0., nran, xran, _pol)
  i = np.nonzero(np.logical_and(x >= e * fran[0], x <= e * fran[1]))[0]
  fit = htanfit(x[i], y[i])
  return x, y, np.array((pol, e)), fit[0]

def winqran(_win, _q = 0, si = 1., _mm = [0, MAXSI32]):
  win = [_win[0], _win[1]]
  mm = [_mm[0], _mm[1]]
  if Type(win[0]) is float: win[0] = int(round(win[0]/si))
  if Type(win[1]) is float: win[1] = int(round(win[1]/si))
  q = _q if Type(_q) is int else int(round(_q/si))
  if Type(mm[0]) is float: mm[0] = int(round(mm[0]/si))
  if Type(mm[1]) is float: mm[1] = int(round(mm[1]/si))
  if not(q): return win
  lo, hi = float(win[0]), float(win[1])
  Q = float(q)
  mid = 0.5 * (lo + hi)
  ran = hi - lo
  ran = max(1., Q * round(ran / Q))
  wid = 0.5*ran
  ran = int(ran)
  win[0] = max(mm[0], int(round(mid - wid)))
  win[1] = win[0] + ran
  if win[1] > mm[1]:
    win[1] = mm[1]
    win[0] = max(mm[0], win[1] - ran[0])
  return win

