import numpy as np
import abfread
import channel

class ABF(abfread.ABF):
  IntData = None
  ADCData = None
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.OpenFile(_fileName)
  def OpenFile(self, _fileName):
    self.ReadFileInfo(_fileName)
  def ReadIntData(self):
    #from scipy.io.numpyio import fread ' now replaced by numpy.fromfile 
    if not(self.nData):
      return np.array('h')
    fh = open(self.fileName, 'rb')
    fh.seek(self.headOffset)
    intdata = np.fromfile(fh, dtype = 'h', count = self.nData)
    fh.close()
    intdata = intdata.reshape( (self.nEpisodes * self.nSamples, self.nChannels) )
    self.IntData = np.empty((self.nChannels, self.nEpisodes * self.nSamples), dtype = np.int16)
    for i in range(self.nChannels):
      self.IntData[i] = intdata[:,i]
    self.IntData = self.IntData.reshape( (self.nChannels, self.nEpisodes, self.nSamples) )
    return self.IntData
  def ReadADCData(self):
    if not(self.nData):
        return np.array('f')
    if self.IntData is None:
        self.ReadIntData()
    self.ADCData = np.empty( (self.nChannels, self.nEpisodes, self.nSamples), dtype = float)
    for i in range(self.nChannels):
        self.ADCData[i,:,:] = self.gain[i] * self.IntData[i] + self.offset[i]
    return self.ADCData    
  def ReadChannelInfo(self):
    chans = [[]] * self.nChannels
    for i in range(self.nChannels):
      chans[i] = channel.chWave()    
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
  def ReadOnsets(self): # doesn't really exist for ABF files
    if not(self.nData):
      return np.array([], dtype = int)
    return np.arange(self.nEpisodes, dtype = int) *  self.nSamples
  def NumberOfChannels(self):
    return self.nChannels
  def NumberOfEpisodes(self):
    return self.nEpisodes
  def NumberOfSamples(self):
    return self.nSamples
  def CloseFile(self):
    pass
