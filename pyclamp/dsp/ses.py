# Wrapper class for sesread.WCP and sesread.EDR to be used with Pyclamp

# Gary Bhumbra

import os
import numpy as np
from pyclamp.dsp.sesread import WCP, EDR
from pyclamp.dsp.channel import chWave

class SES (WCP): # WCP inherits from EDR
  edr = None
  wcp = None 
  IntData = None
  ADCData = None
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.OpenFile(_fileName)
  def OpenFile(self, _fileName):
    _, en = os.path.splitext(_fileName)
    en = en.lower()
    self.wcp = en.lower() == ".wcp" # default to edr if unknown extension
    self.edr = not(self.wcp)
    if self.edr:
      head = EDR.ReadFileInfo(self, _fileName)
      self.nData = self.ndata
    else:
      head = self.ReadFileInfo(_fileName)
    return head
  def ReadIntData(self):
    if not(self.nData):
      return np.array('h')
    if self.edr:
      self.IntData = np.reshape(EDR.ReadIntData(self).T, (self.nChan, self.nEpis, self.nSamp))
      return self.IntData
    fh = open(self.fileName, 'rb')
    self.IntData = np.empty((self.nChan, self.nEpis, self.nSamp), dtype = np.int16)
    for i in range(self.nEpis):
      self.IntData[:, i, :] = self.ReadIntEpis(i, fh).T
    fh.close()
    return self.IntData
  def ReadADCData(self):
    if not(self.nData):
        return np.array('f')
    if self.IntData is None:
        self.ReadIntData()
    self.ADCData = np.empty( (self.nChan, self.nEpis, self.nSamp), dtype = float)
    for i in range(self.nChan):
        self.ADCData[i,:,:] = self.gain[i] * self.IntData[i] + self.offset[i]
    return self.ADCData    
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
    if self.ADCData is not None:
      del self.ADCData
      self.ADCData = None
  def ReadOnsets(self): # don't know if supported by WCP files
    if not(self.nEpis):
      return np.array([], dtype = int)
    return np.arange(self.nEpis, dtype = int) *  self.nSamp
  def NumberOfChannels(self):
    return self.nChan
  def NumberOfEpisodes(self):
    return self.nEpis
  def NumberOfSamples(self):
    return self.nSamp
  def CloseFile(self):
    pass

