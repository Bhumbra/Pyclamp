from iofunc import *
from lsfunc import *
import os
import channel

class TDF:
  readFunc = {'.tab':readDTFile, '.tdf':readDTFile, '.tsv':readSVFile}
  dataFunc = {'.tab':readDTData, '.tdf':readDTData, '.tsv':list}
  writeFunc = {'.tab':writeDTFile, '.tdf':writeDTFile, '.tsv':writeSVFile}
  def __init__(self, _fn = None):
    self.initialise(_fn)
  def initialise(self, _fn = None):
    self.nc = 0
    self.ne = 0
    self.ns = 0
    self.si = 0
    self.oi = -1
    self.io = []
    self.Chan = []
    self.setFn(_fn)
  def setFn(self, _fn = None, _extn = None):
    self.fn = _fn
    self.text = None
    self.Chan = None
    self.data = None
    self.Data = None
    self.extn = _extn
    if self.fn is None: return
    if self.extn is None:
      _, self.extn = os.path.splitext(self.fn)
  def ret_nc(self):
    return self.nc
  def ret_ne(self):
    return self.ne
  def ret_ns(self):
    return self.ns
  def readFile(self, _fn = None, _extn = None):
    if _fn is not None: self.setFn(_fn, _extn)
    self.text = self.readFunc[self.extn](self.fn)
    return self.text
  def readText(self, _fn = None, _extn = None):
    if _fn is not None: self.setFn(_fn, _extn)
    if self.text is None: self.readFile(_fn, _extn)
    self.nc = len(self.text)
    self.io = range(self.nc)
    for i in range(len(self.text)):
      if type(self.text[i]) is tuple:
        self.oi = i
        self.nc -= 1
      else:
        self.ne = len(self.text[i])
        if self.ne:
          self.ns = len(self.text[i][0])
          if self.ns < 4:
            self.ne -= 1
            self.ns = len(self.text[i][-1])
    if self.oi >= 0: del self.io[self.oi]
    self.data = self.dataFunc[self.extn](self.text)
    return self.data
  def readChan(self, _fn = None, _extn = None):
    if _fn is not None: self.setFn(_fn, _extn)
    if self.data is None: self.readText(self.fn, self.extn)
    self.SiGnOf = [[]] * self.nc
    self.Chan = [[]] * self.nc
    h = -1
    for i in self.io:
      h += 1
      self.Chan[h] = channel.chWave()
      chan = self.Chan[h]
      chan.index = h
      chan.name = 'Channel ' + str(h)
      chan.units = 'Units ' + str(h)
      signof = np.array([1., 1., 0.])
      dat = self.data[i]
      ndat = len(dat)
      if ndat:
        Dat = dat[0]
        nDat = len(Dat)
        if not(nDat):
          if self.si: signof[0] = self.si
        else:
          if nDat > 3:
            if self.si: signof[0] = self.si
          else:
            signof[0] = Dat[0]
            if not(self.si): self.si = signof[0]
            if nDat > 1: signof[1] = Dat[1]
            if nDat > 2: signof[2] = Dat[2]
        chan.samplint = signof[0]
        chan.gain = signof[1]
        chan.offset = signof[2]
        self.Chan[h] = chan
        self.SiGnOf[h] = signof.copy()
    return self.Chan
  def readData(self, _fn = None, _extn = None):
    if _fn is not None: self.setFn(_fn, _extn)
    if self.Chan is None: self.readChan(_fn, _extn)
    self.Data = []
    ndata = len(self.data)
    if not(ndata): return self.Data
    I = range(ndata)
    if self.oi >= 0: 
      self.readOnsets()
      I = range(1, ndata)
    self.readWave()
    return self.Data
  def clearData(self, ev = None):
    if self.text is not None:
      del self.text
      self.text = None
    if self.data is not None:
      del self.data
      self.data = None
    if self.Data is not None:
      del self.Data
      self.Data = None
    if self.Chan is not None:
      del self.Chan
      self.Chan = None
    self.initialise()
  def readOnsets(self):
    i = self.oi
    if i < 0:
      self.onsets = np.arange(self.ne)*self.ns if self.ns and self.ne else np.array([], dtype = int)
      return self.onsets
    self.onsets = np.ravel(self.data[i])
    self.si = 0 
    if len(self.onsets):
      _si =  float(self.onsets[0])
      if _si < 0.: # negation of first tuple element is a flag convention for denoting a sampling rate
        self.si = -_si
        self.onsets = np.array(self.onsets[1:], dtype = int)
      elif self.si:
        self.onsets = np.array(self.onsets/self.si, dtype = int)
    return self.onsets
  def readWave(self, _fn = None):
    if self.Chan is None: self.readChan(_fn)
    self.Data = [[]]  * self.nc
    h = -1
    for i in self.io:
      h += 1
      dat = self.data[i]
      ndat = len(dat)
      if ndat:
        Dat = dat[0]
        nDat = len(Dat)
        if nDat:
          if nDat > 3:
            self.Data[h] = np.array(self.data[i])
          else:
            self.Data[h] = self.data[i][1:]
            if nDat > 2: self.Data[h] = np.array(self.Data[h], dtype = int)
    return self.Data
  def retChannels(self):
    return self.Chan
  def setChan(self, _chan = None):    
    chan_ = channel.chWave()
    chan_.index = 0
    chan_.name = '' 
    chan_.units = ''   
    chan_.samplint = None
    chan_.gain = 1
    chan_.offset = 0      
    if type(_chan) is int or type(_chan) is float:
      chan_.samplint = float(_chan)
      chan_.gain = 1
      chan_.offset = 0
    elif type(_chan) is list:
      if len(_chan) > 0: 
        if type(_chan[0]) is int or type(_chan[0]) is float:
          chan_.samplint = float(_chan[0])
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
      if len(_chan) > 1: 
        if type(_chan[1]) is int or type(_chan[1]) is float:
          chan_.gain = _chan[1]
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
      if len(_chan) > 2: 
        if type(_chan[0]) is int or type(_chan[0]) is float:
          chan_.offset = _chan[2]
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
    elif isinstance(_chan, channel.chWave):
      chan_.index = _chan.index
      chan_.name = _chan.name
      chan_.units = _chan.units
      chan_.samplint = float(_chan.samplint) 
      chan_.gain = _chan.gain
      chan_.offset = _chan.offset
    return chan_
  def setData(self, _Data = None, _Chan = None, _onsets = None):
    if _Data is None: return
    if _Chan is None: _Chan = 1
    self.Data = _Data
    self.Chan = []
    self.SiGnOf = []
    self.nc = len(self.Data)
    _Chan = repl(_Chan, self.nc)    
    if not(self.nc): return 
    self.Chan = [[]] * self.nc
    self.SiGnOf = [[]] * self.nc
    for i in range(self.nc):
      chinfo_ = self.setChan(_Chan[i])
      self.si = chinfo_.samplint
      self.Chan[i] = chinfo_
      self.SiGnOf[i] = [chinfo_.samplint, chinfo_.gain, chinfo_.offset]
    self.ne = len(self.Data[0])
    if self.ne:
      self.ns = len(self.Data[0][0])
    self.onsets = np.arange(self.ne, dtype=int)*self.ns if _onsets is None else _onsets
  def writeData(self, _fn = None, onsOpts = 2, wavOpts = 2, ch = None, win = None, ind = None, _extn = None):
    # Opts:
    # 0 = do not export
    # 1 = export floating point data with no headers
    # 2 = export integer data with headers
    if _fn is None: raise ValueError("Export filename mandatory")
    if ch is None: ch = range(self.nc)
    if win is not None:
      nwin = len(win)
      for i in range(nwin):
        if Type(win[i]) is float: win[i] = int(win[i] / self.si)
        win[i] = max(0, min(win[i], self.ns))
    else:
      win = [0, self.ns]
    if ind is None: ind = range(self.ne)
    wind = np.arange(win[0], win[1])
    nX = int(onsOpts != 0) + len(ch)
    X = [[]] * nX
    h = -1
    if onsOpts:
      ons = self.onsets[ind]
      if onsOpts == 1 and self.si:
        ons = np.array(ons, dtype = float) * float(self.si)
      else:
        ons = [-self.si] + np.array(ons, dtype = int).tolist()
      ons = tuple(ons)
      h += 1
      X[h] = ons
    if wavOpts:
      for i in ch:
        h += 1
        signof = self.SiGnOf[i]
        Dat = self.Data[i] if type(self.Data[i]) is np.ndarray else np.array(self.Data[i])
        Dat = Dat[ind, :]
        Dat = Dat[:, wind]
        if wavOpts == 1:
          x = (np.array(Dat, dtype = float) * signof[1] + signof[2]).tolist()
        else:
          x = [signof]
          x += Dat.tolist()
        X[h] = x[:]
    if _extn is None: _, _extn = os.path.splitext(_fn)
    self.writeFunc[_extn](_fn, X)
    return X


