import channel
import numpy as np
from fpfunc import *
from wffunc import *
from types import *

# An active channel modules

class feature(deltaxy): # step and ramp provider
  def rety(self, _x = 0, _i = 0):
    if isnum(_x): _x = [_x]
    x = _x if type(_x) is np.ndarray else np.array(_x)
    x0, y0 = self.retxy(0, _i)
    x1, y1 = self.retxy(1, _i)
    dx, dy = x1-x0, y1-y0
    y = np.zeros(x.shape, dtype = type(y0))
    i = np.nonzero(np.logical_and(x0 <= x, x < x1))[0]
    if not(len(i)): return y
    xi = x[i]
    dx0 = xi - x0
    dx1 = x1 - xi
    if isint(dx):
      if dx == 0: dx == 1
    else:
      if dx == 0.: dx = 1.
    yi = (dx0*y1 + dx1*y0) / dx
    y[i] = yi
    return y

class protocol(channel.chWaev):
  def __init__(self, _index = 0, _name = "", _data = None, _units = "", _quantint = 1., 
               _waev = None, _samplint = None, _gain = None, _offset = None):
    channel.chWaev.__init__(self, _index, _name, _data, _units, _quantint, _waev, _samplint, _gain, _offset)
    self.setFeature()
  def initProt(self, _ns = 0, _ne = 1, _nl = 0):
    self.ns = int(_ns)
    self.ne = int(_ne)
    self.nl = int(_nl)
    self.setData(np.arange(self.ne, dtype = int) * (self.ns+self.nl), self.units)
  def setFeature(self, _Feature = None):
    self.Feature = [] if _Feature is None else _Feature
  def Conv2Int(self, _x = 0, _y = 0, _dx = 0, _dy = 0, _Dx = 0, _Dy = 0):
    _x, _y   = self.conv2int(_x , 0), self.conv2int(_y , 1)
    _dx, _dy = self.conv2int(_dx, 0), self.conv2int(_dy, 1)
    _Dx, _Dy = self.conv2int(_Dx, 0), self.conv2int(_Dy, 1)
    return _x, _y, _dx, _dy, _Dx, _Dy
  def addFeature(self, _x = 0, _y = 0, _dx = 0, _dy = 0, _Dx = 0, _Dy = 0):
    _x, _y, _dx, _dy, _Dx, _Dy = self.Conv2Int(_x, _y, _dx, _dy, _Dx, _Dy)
    self.Feature.append(feature(_x, _y, _dx, _dy, _Dx, _Dy))
  def addRamp(self, _x0 = 0, _y0 = 0, _dx = 0, _y1 = 0, _Dx = 0, _Dy = 0):
    self.addFeature(_x0, _y0, _dx, _y1-_y0, _Dx, _Dy)
  def addStep(self, _x0 = 0, _y0 = 0, _dx = 0, _Dx = 0, _Dy = 0):
    self.addFeature(_x0, _y0, _dx, 0, _Dx, _Dy)
  def rety(self, _x = 0, _i = 0, _y = 0):
    if isnum(_x): _x = [_x]
    _x = _x if type(_x) is np.ndarray else np.array(_x)
    x = self.conv2int(_x, 0)
    _y = np.array(_y, dtype = self.Dtype)
    y = np.tile(_y, x.shape)
    for _feature in self.Feature:
      y += _feature.rety(x, _i)
    return y
  def estProt(self, _Y, ny0 = 256, z = 2): # z = number of sweeps to look at 
    # currently only does ascending/descending/stationary single steps in same time window using only first 2 episodes
    if nDim(_Y) == 2:
      Y = _Y
      while len(Y) < z:
        Y = np.hstack( (Y, Y[-1].reshape((1,len(Y[-1])))) )
    else:
      Y = np.tile(Y.reshape((1, len(_Y))), (z, 1))
    self.initProt(len(Y[0]), len(Y))
    self.setFeature()
    ny0 = min(ny0, self.ns)
    Yz = Y[:z]
    Y0 = Yz[:,:ny0]
    Y0min, Y0max, Y0med = Y0.min(axis=1), Y0.max(axis=1), np.median(Y0, axis=1)
    Yzmin, Yzmax = Yz.min(axis=1), Yz.max(axis=1)
    Ypos = Yzmax - Y0med >= Y0med - Yzmin
    ipos = np.nonzero(Ypos)[0]
    ineg = np.nonzero(np.logical_not(Ypos))[0]
    Yh = np.empty(z, dtype = float)
    Yh[ipos] = 0.5*(Y0med[ipos] + Yzmax[ipos])
    Yh[ineg] = 0.5*(Y0med[ineg] + Yzmin[ineg])
    D = np.zeros(z, dtype = float)
    I = np.zeros(z, dtype = int)
    J = np.tile(self.ns, z)
    for i in range(z):
      More = None
      if Ypos[i]:
        if Yh[i] > Y0max[i]:
          More = np.greater
      else:
        if Yh[i] < Y0min[i]:
          More = np.less
      if More is not None:
        y = Yz[i]
        ind = np.nonzero(More(y, Yh[i]))[0]
        if len(ind):
          D[i] = np.median(y[ind]) - Y0med[i]
          I[i] = ind[0]
          J[i] = ind[-1]
    h = np.argmax(np.fabs(D))
    i, j = I[h], min(J[h]+1, self.ns)
    k = int(round(0.5*float(i+j)))
    x0 = i
    dx = j - i
    y_ = Y0med[0]
    y0 = D[h]
    Dy = D[1] - D[0]
    self.addStep(x0, y0, dx, 0, Dy)
    return y_, x0, y0, dx, Dy
 
