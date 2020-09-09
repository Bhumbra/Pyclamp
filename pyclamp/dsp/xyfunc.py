# A module to perform GUI-tasks outside a GUI

import numpy as np
from dtypes import *
from fpfunc import *

def rescale(x, xlo, xhi, Xlo = 0., Xhi = 1.):
  dx = xhi - xlo
  if dx == 0.: return x
  return Xlo + (Xhi - Xlo) * (x - xlo) / dx

class xygui:
  X = None
  Y = None
  x = None
  y = None
  rx = None
  ry = None
  ellx = None
  elly = None
  xyabpsz = None
  maxrd = 0.025 # minimum relative distance to grab
  dragtype = 0 # 0 = No drag, 1 = Ellipse, 2 = Major Axis, 3 = Minor axis
  def __init__(self, _X = [], _Y = []):
    self.xm = [0., 1.]                                  # xlimits
    self.ym = [0., 1.]                                  # ylimits
    self.x4 = None
    self.y4 = None
    self.setData(_X, _Y)
    self.defEllipse()
  def setData(self, _X = [], _Y = [], _f = 0):
    _NX, _NY = len(_X), len(_Y)
    if _NX != _NY:
      raise ValueError("(X, Y) inputs incommensurate.")
    self.iniDims(_NX)
    if not(self.N): return
    if not(isarray(_X[0])) and not(isarray(_Y[0])): _X, _Y = [_X], [_Y]
    self.iniDims(len(_X))  
    for i in range(self.N):
      self.n[i] = len(_X[i])
      if len(_Y[i]) != self.n[i]:
        raise ValueError("(X[" + str(i) + "], Y[" + str(i) + "] inputs incommensurate.")
      self.X[i] = np.array(_X[i], dtype = float)
      self.Y[i] = np.array(_Y[i], dtype = float)
    self.setFocus(self.N - 1)
  def addData(self, _x = [], _y = [], _f = None):
    if _f == True: _f = self.f
    if _f is None: _f = self.N
    self.f = _f
    self.x = np.array(_x, dtype = float)
    self.y = np.array(_y, dtype = float)
    self.n = np.hstack( (self.n, len(self.x)) )
    if self.n[-1] != len(self.y):
      raise ValueError("(x, y) inputs incommensurate.")
    self.N += 1
    if self.N == 1:
      self.X = [np.copy(self.x)]
      self.Y = [np.copy(self.y)]
    else:
      self.X.append(self.x)
      self.Y.append(self.y)
    return self.N - 1
  def iniDims(self, _N):
    self.N = _N
    self.n = np.zeros(self.N, dtype = int)
    self.X = [None] * self.N
    self.Y = [None] * self.N
  def setFocus(self, _f = 0):
    self.f = _f
    self.x = self.X[self.f] 
    self.y = self.Y[self.f] 
    self.nf = self.n[self.f]
    self.xm = [self.x.min(), self.x.max()]
    self.ym = [self.y.min(), self.y.max()]
    self.xmid = 0.5 * (self.xm[1] + self.xm[0])
    self.xwid = 0.5 * (self.xm[1] - self.xm[0])
    self.ymid = 0.5 * (self.ym[1] + self.ym[0])
    self.ywid = 0.5 * (self.ym[1] - self.ym[0])
    return self.x, self.y
  def argnear(self, _x = 0., _y = 0., ok = True, xx = None, yy = None):
    if xx is None: xx = self.xm
    if yy is None: yy = self.ym
    if type(ok) is bool: ok = np.tile(ok, self.nf)
    i = argtrue(ok) 
    xi = self.x[i]
    yi = self.y[i]
    if isarray(_x) and isarray(_y): # range searches are actually easier
      if len(_x) == 2 and len(_y) == 2:
        xr, yr = np.sort(_x), np.sort(_y)
        xo = np.logical_and(xi >= xr[0], xi <= xr[1]) 
        yo = np.logical_and(yi >= yr[0], yi <= yr[1]) 
        I = argtrue(np.logical_and(xo, yo))
        if len(I): return i[I]
      _x, _y = xr.mean(), yr.mean()
    self.rx = rescale(xi, xx[0], xx[1])
    self.ry = rescale(yi, yy[0], yy[1])
    dx = self.rx - rescale(_x, xx[0], xx[1])
    dy = self.ry - rescale(_y, yy[0], yy[1])
    r2 = dx**2. + dy**2.
    return i[np.argmin(r2)]
  def oldargnear(self, _x = 0., _y = 0., ok = True, xx = None, yy = None):
    if type(ok) is bool: ok = np.tile(ok, self.nf)
    if xx is None: xx = self.xm
    if yy is None: yy = self.ym
    i = np.nonzero(ok)[0]
    xi = self.x[i]
    yi = self.y[i]
    self.rx = rescale(xi, xx[0], xx[1])
    self.ry = rescale(yi, yy[0], yy[1])
    dx = self.rx - rescale(_x, xx[0], xx[1])
    dy = self.ry - rescale(_y, yy[0], yy[1])
    r2 = dx**2. + dy**2.
    return i[np.argmin(r2)]
  def relDist(self, x0, y0, x1 = None, y1 = None, dx = None, dy = None):
    if x1 is None: x1 = self.xmid
    if y1 is None: y1 = self.ymid
    if dx is None: dx = self.xwid
    if dy is None: dy = self.ywid
    return np.sqrt( ((x0 - x1)/unzero(dx))**2. + ((y0 - y1)/unzero(dy))**2. )
  def defEllipse(self, _w = 400, _z = 0.5, _xyabps = None):
    if _xyabps is None:
      self.xyabps = np.empty(6, dtype = float)
      self.xyabps[0] = self.xmid
      self.xyabps[1] = self.ymid
      self.xyabps[2] = self.xwid
      self.xyabps[3] = self.ywid
      self.xyabps[4] = 0.;
      self.xyabps[5] = 1.;
    else:
      self.xyabps = np.array(listcomplete(_xyabps, 
                    [self.xmid, self.ymid, self.xwid, self.ywid, 0., 1]))
    return self.calcEllipse(_w, _z, self.xyabps)
  def calcEllipse(self, _w = None, _z = None, _xyabps = None):
    self.retEllipse(_w, _z, _xyabps)
    self.retEllipse2()
    self.inEllipse()
    return self.ellx, self.elly, self.x2, self.y2
  def inEllipse(self):
    _w, _z = cart2normel(self.x, self.y, self.xyabps)
    self.inellipse = _z <= self.z
    return self.inEllipse
  def retEllipse(self, _w = None,  _z = None, _xyabps = None):
    if _w is None:
      _w = self.w
    else:
      if type(_w) is int: _w = np.linspace(-np.pi, np.pi, _w)
      self.w = np.copy(_w)
    if _z is None:
      _z = self.z
    else:
      self.z = _z
    if _xyabps is None:
      _xyabps = self.xyabps
    else:
      self.xyabps = np.array(listcomplete(_xyabps, [0., 0., 1., 1., 0., 1]))
    self.ellx, self.elly = normel2cart(self.w, self.z, self.xyabps)
    return self.ellx, self.elly
  def retEllipse2(self, _xm = None, _ym = None):
    if _xm is None: _xm = self.xm
    if _ym is None: _ym = self.ym
    mx = 0.5 * (_xm[0] + _xm[1])
    my = 0.5 * (_ym[0] + _ym[1])
    w4 = np.linspace(-np.pi, np.pi, 5)[:-1]
    self.x4, self.y4 = normel2cart(w4, self.z, self.xyabps)
    z4 = self.relDist(self.x4, self.y4)
    self.x2 = np.empty(2, dtype = float)
    self.y2 = np.empty(2, dtype = float)
    if z4[0] < z4[2]:
      self.x2[0], self.y2[0] = self.x4[0], self.y4[0]
    else:  
      self.x2[0], self.y2[0] = self.x4[2], self.y4[2]
    if z4[1] <= z4[3]:
      self.x2[1], self.y2[1] = self.x4[1], self.y4[1]
    else:  
      self.x2[1], self.y2[1] = self.x4[3], self.y4[3]
    return self.x2, self.y2
  def dragEllipse(self, status, X, Y):
    if status == 2: # 1 = button down, 2 = drag, 3 = button up
      if self.dragtype == 1:
        self.xyabps[0] = self.xyabpsz[0] + X - self.X0
        self.xyabps[1] = self.xyabpsz[1] + Y - self.Y0    
        return self.calcEllipse()
      if self.dragtype == 2:
        dx = (X - self.xyabps[0]) / self.xyabps[2]
        dy = (Y - self.xyabps[1]) / self.xyabps[3]
        self.z = np.sqrt(dx**2 + dy**2)
        self.xyabps[4] = atan2(dx, dy)
        self.xyabps[5] = self.xyabpsz[5] * self.xyabpsz[6] / unzero(self.z)
        return self.calcEllipse()
    if status == 1:
      self.X0, self.Y0 = X, Y
      self.xyabpsz = np.hstack((self.xyabps, self.z))
      if self.x4 is None: self.retEllipse2()
      rd = self.relDist(X, Y, self.x2, self.y2) 
      if rd[0] <= self.maxrd:
        self.dragtype = 2
        return True
      if rd[1] <= self.maxrd:
        self.dragtype = 3
        # now invert major and minor axis
        dx = (self.x2[1] - self.xyabps[0]) / unzero(self.xyabps[2])
        dy = (self.y2[1] - self.xyabps[1]) / unzero(self.xyabps[3])
        self.z = np.sqrt(dx**2. + dy**2.)
        dx = (self.x2[0] - self.xyabps[0]) / unzero(self.xyabps[2])
        dy = (self.y2[0] - self.xyabps[1]) / unzero(self.xyabps[3])
        self.xyabps[4] += 0.5*np.pi
        self.xyabps[5] = np.sqrt(dx**2. + dy**2.) / unzero(self.z)
        self.xyabpsz = np.hstack((self.xyabps, self.z))
        self.dragtype = 2
        return True
      rd = self.relDist(X, Y, self.ellx, self.elly)
      if rd.min() <= self.maxrd:
        self.dragtype = 1
        return True
      return False
    if status == 0:
      if self.dragtype == 1:
        self.xyabps[0] = self.xyabpsz[0] + X - self.X0
        self.xyabps[1] = self.xyabpsz[1] + Y - self.Y0    
        self.dragtype = 0
        return self.calcEllipse()
      if self.dragtype == 2:
        dx = (X - self.xyabps[0]) / unzero(self.xyabps[2])
        dy = (Y - self.xyabps[1]) / unzero(self.xyabps[3])
        self.z = np.sqrt(dx**2 + dy**2)
        self.xyabps[4] = atan2(dx, dy)
        self.xyabps[5] = self.xyabpsz[5] * self.xyabpsz[6] / unzero(self.z)
        self.dragtype = 0
        return self.calcEllipse()

