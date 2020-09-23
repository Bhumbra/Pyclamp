# Wrapper class for smrread.SMR to be used with Pyclamp

# Gary Bhumbra

import numpy as np
from pyclamp.dsp.smrread import SMR
from pyclamp.dsp.channel import chWave

SMR_CHANNEL_SELECT_RULE = None # None = Auto, True = By max. number of episode, False = By max number of samples
# If Auto - it's based on extension capitalisation!

class SMR(SMR):
  IntData = None
  ADCData = None
  nChan = 0
  index = None
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.OpenFile(_fileName)
  def OpenFile(self, _fileName):
    self.ReadFileInfo(_fileName)
    self.smr_channel_select_rule = SMR_CHANNEL_SELECT_RULE
    if self.smr_channel_select_rule is None:
      self.smr_channel_select_rule = self.fileName.find('.SMR') < 0
    if not(self.nchan): return 
    self.nepis = [0] * self.nchan
    self.nsamp = [0] * self.nchan
    self.gain = np.ones(self.nchan)
    self.offset = np.zeros(self.nchan)
    for i in range(self.nchan):
      if self.kinds[i] == 1:
        self.nepis[i] = 1
        self.nsamp[i] = self.ndata[i]
      elif self.kinds[i] == 6:
        self.nepis[i] = self.ndata[i]
        self.nsamp[i] = self.channelHeaders[i].nExtra/2
      if self.nepis[i] > 0:
        self.gain[i] = self.channelHeaders[i].scale / 6553.6
        self.offset[i] = self.channelHeaders[i].offset
    i = np.argmax(self.nepis) if self.smr_channel_select_rule else np.argmax(self.nsamp)
    self.nEpis = self.nepis[i]
    self.nSamp = self.nsamp[i]
    self.index = np.nonzero(np.logical_and(np.array(self.nepis) == self.nEpis, np.array(self.nsamp) == self.nSamp))[0]
    self.nChan = len(self.index)
    self.gain = self.gain[self.index]
    self.offset = self.offset[self.index]
  def ReadIntData(self, _fileName = None, _chan = None):
    if _chan is not None: return smrread.SMR.ReadIntData(self, None, _chan)
    if self.index is None: self.OpenFile(_fileName)
    if self.index is None: return
    return smrread.SMR.ReadIntData(self, None, list(self.index))
  def ReadADCData(self, _fileName = None):
    if not(self.nData):
        return np.array('f')
    if self.IntData is None:
        self.ReadIntData(_fileName)
    if self.IntData is None: return
    self.ADCData = np.empty( (self.nChan, self.nEpis, self.nSamp), dtype = float)
    for i in range(self.nChan):
        self.ADCData[i,:,:] = self.gain[i] * self.IntData[i] + self.offset[i]
    return self.ADCData    
  def ReadChannelInfo(self):
    chans = [[]] * self.nChan
    for i in range(self.nChan):
      chans[i] = chWave()    
      chans[i].index = i # self.index[i]      
      chans[i].name = self.channelHeaders[self.index[i]].title
      chans[i].units = self.channelHeaders[self.index[i]].units
      chans[i].samplint = self.fileHeader.dTimeBase
      chans[i].gain = self.gain[i]
      chans[i].offset = self.offset[i]
    return chans    
  def ClearData(self):
    if self.Indices is not None:
      del self.Indices
      self.Indices = None
    if self.Markers is not None:
      del self.Markers
      self.Markers = None
    if self.IntData is not None:
      del self.IntData
      self.IntData = None
    if self.ADCData is not None:
      del self.ADCData
      self.ADCData = None
  def ReadOnsets(self):
    i = self.index[0]
    if self.kinds[i] != 6:
      return np.arange(self.nEpis, dtype = int) *  self.nSamp
    return np.array(self.Indices[0], dtype = int)
  def NumberOfChannels(self):
    return self.nChan
  def NumberOfEpisodes(self):
    return self.nEpis
  def NumberOfSamples(self):
    return self.nSamp
  def CloseFile(self):
    pass

