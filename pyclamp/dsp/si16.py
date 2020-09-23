import numpy as np
from pyclamp.dsp.channel import *

def si16read(fn, datatype = 'h'):
  f = open(fn, 'rb')
  Data = np.fromfile(file=f, dtype = datatype)
  f.close();
  return Data

class SI16: # a fairly primitive short-data binary type reader
  IntData = None
  fileName = None
  chinfo = None
  nChan = 0 # number of channels
  nEpis = 0 # number of episodes
  nSamp = 0 # number of samples per episode (per channel)
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.OpenFile(_fileName)
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.ReadFileInfo(_fileName)
  def ReadFileInfo(self, _fileName = None):
    if _fileName == None: return
    self.fileName = _fileName
    fh = open(self.fileName, mode = 'rb')
    fh.seek(0, 2)
    siz = fh.tell()
    fh.close()
    self.nSamp = int(0.5*float(siz))
    if self.nSamp:
      self.nChan = 1
      self.nEpis = 1
      self.gain = [1.]
      self.offset = [0.]
      self.samplint = 1.
      self.name = ["SI16 channel"]
      self.units = ["units"]
    return self
  def OpenFile(self, _fileName):
    self.ReadFileInfo(_fileName)
  def ReadIntData(self, datatype='h'):
    if not(self.nSamp):
      return np.array('h')
    fh = open(self.fileName, 'rb')
    self.IntData = np.fromfile(file=fh, dtype = datatype)
    fh.close();
    if datatype == 'h': self.IntData = np.array([np.array([self.IntData[:self.nSamp]])])
    return self.IntData
  def ReadChannelInfo(self):
    chans = [[]] * self.nChan
    for i in range(self.nChan):
      chans[i] = chWave()    
      chans[i].index = i            
      chans[i].name = self.name[i]
      chans[i].units = self.units[i]
      chans[i].samplint = self.samplint
      chans[i].gain = self.gain[i]
      chans[i].offset = self.offset[i]
    return chans    
  def ClearData(self):
    if self.IntData is not None:
      del self.IntData
      self.IntData = None
  def ReadOnsets(self): # don't know if supported by WCP files
    if not(self.nSamp):
      return np.array([], dtype = int)
    return np.array([0], dtype = int)
  def NumberOfChannels(self):
    return self.nChan
  def NumberOfEpisodes(self):
    return self.nEpis
  def NumberOfSamples(self):
    return self.nSamp
  def CloseFile(self):
    pass

