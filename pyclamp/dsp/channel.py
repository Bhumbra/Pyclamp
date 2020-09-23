# A signal channel module

# Gary Bhumbra

import numpy as np
from pyclamp.dsp.dtypes import *

#-------------------------------------------------------------------------------
class chInfo: # Basic channel info
  def __init__(self, _index = 0, _name = ""):
    self.setInfo(_index, _name)
  def setInfo(self, _index = 0, _name = ""):
    self.index = _index
    self.name = _name

#-------------------------------------------------------------------------------
class chData: # Basic channel data
  def __init__(self, _data = None, _units = ""):
    self.setData(_data, _units)
  def setData(self, _data = None, _units = ""):
    self.data = _data
    self.units = _units
    self.dtype = elType(self.data)

#-------------------------------------------------------------------------------
class chBase (chInfo, chData): # Base channel class
  def __init__(self, _index = 0, _name = "", _data = None, _units = ""):
    chInfo.__init__(self, _index, _name)
    chData.__init__(self, _data, _units)

#-------------------------------------------------------------------------------
class chTime (chBase): # Time class
  time = None
  def __init__(self, _index = 0, _name = "", _data = None, _units = "", _quantint = 1.):
    chInfo.__init__(self, _index, _name)
    chData.__init__(self, _data, _units)
    self.setQuan(_quantint)
  def setQuan(self, _quantint = 1.):
    self.quantint = _quantint
    if self.data is None: return
    self.time = self.data if self.dtype is float and self.quantint == 1. else self.data * self.quantint

#-------------------------------------------------------------------------------
class chEvnt (chTime): # Event class
  evntclass = chData
  def __init__(self, _index = 0, _name = "", _data = None, _units = "", _quantint = 1., _evnt = None):
    chTime.__init__(self, _index, _name, _data, _units, _quantint)
    self.setevnt(_evnt)
  def setevnt(self, _evnt = None):
    if isinstance(_evnt, self.evntclass):
      self.evnt = _evnt
    else:
      self.evnt = self.evntclass(_evnt, self.units)
    return self.evnt

#-------------------------------------------------------------------------------
class chWave (chBase): # Waveform class
  def __init__(self, _index = 0, _name = "", _data = None, _units = "", _samplint = 1., _gain = None, _offset = None):
    chInfo.__init__(self, _index, _name)
    chData.__init__(self, _data, _units)
    self.setSamp(_samplint, _gain, _offset)
  def setSamp(self, _samplint = None, _gain = None, _offset = None):
    self.samplint = 1. if _samplint is None else _samplint
    self.gain = 1. if _gain is None else _gain
    self.offset = 0. if _offset is None else _offset
    if isarray(_samplint):
      self.samplint = _samplint[:]
      if len(_samplint) > 0: self.samplint = _samplint[0]
      if len(_samplint) > 1 and _gain is None: self.gain = _samplint[1]
      if len(_samplint) > 2 and _offset is None: self.offset = _samplint[2]
    self.samplint = float(self.samplint)
    self.gain = float(self.gain)
    self.offset = float(self.offset)
  def time2ind(self, t):
    return int(t / self.samplint)
  def ind2time(self, i):
    return float(i * self.samplint)
  def raw2real(self, _w):
    w = float(_w) if isnum(_w) else np.array(_w, dtype = float)
    if self.gain != 1.: w *= self.gain
    if self.offset != 0.: w += self.offset
    return w
  def real2raw(self, _w):
    w = float(_w) if isnum(_w) else np.array(_w, dtype = float)
    if self.offset != 0.: w -= self.offset
    if self.gain != 1.: w /= self.gain
    if isnum(_w):
      return int(w)
    return np.array(_w, dtype = int)
  def conv2int(self, v, ax = 0): #ax = 0 for time, ax = 1 for wave
    xy = [self.time2ind, self.real2raw]
    if isnum(v):
      if isint(v): return v
      return xy[ax](v)
    if elType(v) is int:
      return v
    return xy[ax](v)

#-------------------------------------------------------------------------------
class chWaev (chEvnt): # Wave Event class
  evntclass = chWave
  def __init__(self, _index = 0, _name = "", _data = None, _units = "", _quantint = 1., 
               _waev = None, _samplint = None, _gain = None, _offset = None):
    chEvnt.__init__(self, _index, _name, _data, _units, _quantint)
    self.setwaev(_waev, _samplint, _gain, _offset)
  def setwaev(self, _waev = None, _samplint = None, _gain = None, _offset = None):
    if isinstance(_waev, self.evntclass):
      self.waev = _waev
    else:
      self.waev = self.evntclass(self.index, self.name, _waev, self.units)
    self.setSamp(_samplint, _gain, _offset)
  def setSamp(self, _samplint = None, _gain = None, _offset = None):
    self.waev.setSamp(_samplint, _gain, _offset)
    self.Data = self.waev.data
    self.Units = self.waev.units
    self.Dtype = self.waev.dtype
    self.samplint = self.waev.samplint
    self.gain = self.waev.gain
    self.offset = self.waev.offset
  def time2ind(self, t):
    return self.waev.time2ind(t)
  def ind2time(self, i):
    return self.waev.ind2time(i)
  def raw2real(self, _w):
    return self.waev.raw2real(_w)
  def real2raw(self, _w):
    return self.waev.real2raw(_w)
  def conv2int(self, v, ax = 0): 
    return self.waev.conv2int(v, ax)
    
#-------------------------------------------------------------------------------

