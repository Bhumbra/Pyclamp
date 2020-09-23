# EDR/WCP File reader (continuous/episodic data)

# From the Appendicis in John's manuals.
#
# It appears a WinEDR file is a mixture between MSDOS ASCII text containing the `header block' following by ADC data in
# little-endian (thankfully) signed integer 16-bit binary ADC data separated by channel.

# It seems WinWCP adopts a similar format except ths MSDOS ASCII header is followed by sections of `analysis' then ADC 
# data partitioned by episode then by sample then by channel.  

# Gary Bhumbra

import numpy as np
import struct as st
 
WCP_MAXIMUM_NUMBER_OF_CHANNELS = 128   # theoretical maximum (it's 12 for EDR)
WCP_SECTOR_SIZE_BYTES = 512            # number of bytes in each sector

class ChannelInfo:  # (See John's Appendix)
  # Common to EDR and WCP
  YN = None         # Name
  YU = None         # Units
  YO = None         # Offset into sample groupo in data black
  YZ = None         # Offset (unit)
  # Specific to EDR
  YCF = None        # Callibration factor (V/units)
  YAG = None        # Gain factor
  # Specific to WCP
  YG = None         # Gain (mV/unit)
  YR = None         # No idea 
  def __init__(self, fh = None, offset = 0, channeloffset = None):
    if fh is not None and channeloffset is not None:
      self.readInfo(fh, offset, channeloffset)
  def readInfo(self, fh = None, offset = 0, channeloffset = None):
    fh.seek(offset)
    keyval = None
    for i in range(channeloffset):
      keyval = fh.readline()
    chan = None
    done = False
    while not(done):
      keyval = fh.readline()
      try:
        key, val = keyval.split('=')
      except ValueError:
        done = True
      if not(done):
        if chan is None: chan = key[-1]
        if key[-1] == chan:
          key = key[:-1]
          if key == 'YN': self.YN = val[:-2]
          if key == 'YU': self.YU = val[:-2]
          if key == 'YO': self.YO = int(val)
          if key == 'YZ': self.YZ = int(val)
          if key == 'YCF': self.YCF = float(val)
          if key == 'YAG': self.YAG = float(val)
          if key == 'YG': self.YG = float(val)
          if key == 'YR': self.YR = int(val)

class RecordAnalysisBlock: 
  # AFAICS, the only useful thing here is to read in MaxVoltage within the first episode to calculate gains
  RecordStatus = None
  RecordType = None
  GroupNumber = None
  TimeRecorded = None
  SamplingInterval = None
  MaxVoltage = None
  Marker = None
  def __init__(self, fh = None, offset = 0, nchans = 1):
    if fh is not None:
      self.readInfo(fh, offset, nchans)
  def readInfo(self, fh = None, offset = 0, nchans = 1):
    fh.seek(offset)
    self.RecordStatus = fh.read(8)
    self.RecordType = fh.read(4)
    self.GroupNumber = st.unpack('f', fh.read(4))[0]
    self.TimeRecorded = st.unpack('f', fh.read(4))[0]
    self.SamplingInterval = st.unpack('f', fh.read(4))[0]
    # self.MaxVoltage = [st.unpack('f', fh.read(4))[0] for i in range(nchans)]
    self.MaxVoltage = [[]] * nchans
    for i in range(nchans):
      self.MaxVoltage[i] = st.unpack('f', fh.read(4))[0]
    self.Marker = fh.read(16)

class HeaderBlock:  # (See John's Appendicis) - everything in WCP header includes EDR header info
  VER = None        # Version
  CTIME = None      # Creation date/time
  RTIME = None      # Recording date/time
  RTIMESECS = None  # Time to start of recording since last boot
  NC = None         # Number of channels
  NR = None         # Number of episodes
  NBH = None        # Number of sectors in file header block
  NBA = None        # Number of sectors in record analysis block
  NBD = None        # Number of sectors in record data block (apparently per episode)
  AD = None         # A/D converter input voltage range
  ADCMAX = None     # Maximum ADC sample value
  NP = None         # Number of ADC samples per channel (apparently per episode)
  DT = None         # ADC sampling interval
  NZ = None         # Number of samples averaged to calculate a zero value
  TU = None         # Time units
  ID = None         # Experimental identification line
  HBsize = None     # Size of Header Block
  RAB = None        # Record Analysis Block
  Chan = None       # Channel Info (see above)
  def __init__(self, fh = None, offset = 0):
    if fh is not None:
      self.readInfo(fh, offset)
  def readInfo(self, fh = None, offset = 0):
    keys = []
    vals = []
    fh.seek(offset)
    done = False
    chanOffset = np.tile(32768, WCP_MAXIMUM_NUMBER_OF_CHANNELS)
    i = -1
    while not(done):
      i += 1
      keyval = fh.readline()
      try:
        key, val = keyval.split('=')
      except ValueError:
        done = True
      if not(done):
        chan = None
        try:
          _chan = int(key[-1])
          chan = _chan
        except ValueError:
          pass
        if chan is not None:
          chanOffset[chan] = min(chanOffset[chan], i)
        else: 
          # structure-style class members are more user-friendly than dictionaries if more laborious 
          if key == 'VER': self.VER = float(val)
          if key == 'CTIME': self.CTIME = val[:-2] # exclude MSDOS's LCFR ASCII characters
          if key == 'RTIME': self.RTIME = val[:-2]
          if key == 'RTIMESECS': self.RTIMESECS = float(val)
          if key == 'NC': self.NC = int(val)
          if key == 'NR': self.NR = int(val)
          if key == 'NBH': self.NBH = int(val)
          if key == 'NBA': self.NBA = int(val)
          if key == 'NBD': self.NBD = int(val)
          if key == 'AD': self.AD = float(val)
          if key == 'ADCMAX': self.ADCMAX = int(val)
          if key == 'NP': self.NP = int(val)
          if key == 'DT': self.DT = float(val)
          if key == 'NZ': self.NZ = int(val)
          if key == 'TU': self.TU = val[:-2]
          if key == 'ID': self.ID = val[:-2]
    if self.NC is None: return
    chanOffset = chanOffset[:self.NC]
    self.Chan = [None] * self.NC
    for i in range(self.NC):
      self.Chan[i] = ChannelInfo(fh, offset, chanOffset[i])
    # The formula below is from John's Appendix which confusingly has one unmatched closing parenthesis
    self.HBsize = (int((self.NC - 1)/8) + 1) * 1024
    self.RAB = RecordAnalysisBlock(fh, self.HBsize, self.NC)
    if self.NP is not None: return # It appears NP values are conspicuous by their absence in WCP headers.
    # So let's try to work it out from self.NBD and self.NC if available otherwise we're then stuffed.
    if not(self.NBD) or not(self.NC): return
    self.NP = self.NBD*WCP_SECTOR_SIZE_BYTES/2/self.NC

class EDR:  # WCP reader class for the ADC data 
  fileName = None
  header = None
  chinfo = None
  nChan = 0 # number of channels
  nEpis = 0 # number of episodes (which is always 1 for EDR files)
  nSamp = 0 # number of samples per episode (per channel)
  ndata = 0 # number of samples per episode (all channels)
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.ReadFileInfo(_fileName)
  def ReadFileInfo(self, _fileName = None):
    if _fileName == None: return
    self.fileName = _fileName
    fh = open(self.fileName, mode = 'rb')
    self.header = HeaderBlock(fh)
    self.chinfo = self.header.Chan
    fh.close()
    self.nChan = 0            
    self.nEpis = 0
    self.nSamp = 0
    self.ndata = 0
    self.dataOffset = 0
    self.samplint = []
    self.name = []
    self.units = []
    self.gain = []
    self.offset = []
    if self.header.NP is not None: # and self.header.NR is not None:
      # demystify NC and NP
      self.nChan = self.header.NC                      
      self.nEpis = 1
      self.nSamp = self.header.NP
      self.ndata = self.nChan * self.nSamp
      self.dataOffset = self.header.NBH
      # demystify YXn 
      if self.header.NC is not None:    
        self.samplint = self.header.DT # it appears to be identical for all channels
        self.name = [[]] * self.header.NC
        self.units = [[]] * self.header.NC
        self.gain = [[]] * self.header.NC
        self.offset = [[]] * self.header.NC
        for i in range(self.header.NC):
          self.name[i] = self.chinfo[i].YN
          self.units[i] = self.chinfo[i].YU
          # This formula is from John's Appendix - let's plug and pray
          self.gain[i] = float(self.header.AD) /( (float(self.header.ADCMAX) + 1.) * float(self.chinfo[i].YCF) * float(self.chinfo[i].YAG))
          self.offset[i] = -float(self.chinfo[i].YZ)
    return self.header
  def ReadIntData(self, _fileName = None): # output dimensionality: [episode, sample, channel]
    # Cannot be read in one go because data and analysis appear to alternate across episodes
    if _fileName is not None: self.ReadFileInfo(_fileName)
    if not(self.ndata):
      return np.array('h')
    fh = open(self.fileName, 'rb')
    fh.seek(self.dataOffset)
    intdata = np.fromfile(fh, dtype = 'h', count = self.ndata)
    fh.close()
    self.IntData = np.reshape(intdata, (self.nSamp, self.nChan))
    return self.IntData 

class WCP (EDR):  # WCP reader class for the ADC data 
  nData = 0 # number of samples (all channels)
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.ReadFileInfo(_fileName)
  def ReadFileInfo(self, _fileName = None):
    if _fileName == None: return
    self.fileName = _fileName
    fh = open(self.fileName, mode = 'rb')
    self.header = HeaderBlock(fh)
    self.chinfo = self.header.Chan
    fh.close()
    self.nChan = 0            
    self.nEpis = 0
    self.nSamp = 0
    self.ndata = 0
    self.nData = 0
    self.dataOffset = 0
    self.samplint = []
    self.name = []
    self.units = []
    self.gain = []
    self.offset = []
    if self.header.NP is not None and self.header.NR is not None:
      # demystify NC, NR, and NP
      self.nChan = self.header.NC                      
      self.nEpis = self.header.NR
      self.nSamp = self.header.NP
      self.ndata = self.nChan * self.nSamp
      self.nData = self.ndata * self.nEpis
      self.dataOffset = self.header.HBsize +  WCP_SECTOR_SIZE_BYTES*self.header.NBA
      # demystify YXn 
      if self.header.NC is not None:    
        self.samplint = self.header.DT # it appears to be identical for all channels
        self.name = [[]] * self.header.NC
        self.units = [[]] * self.header.NC
        self.gain = [[]] * self.header.NC
        self.offset = [[]] * self.header.NC
        for i in range(self.header.NC):
          self.name[i] = self.chinfo[i].YN
          self.units[i] = self.chinfo[i].YU
          # This formula is from John's Appendix - let's plug and pray
          self.gain[i] = float(self.header.RAB.MaxVoltage[i]) / (float(self.header.ADCMAX) * float(self.chinfo[i].YG))
          self.offset[i] = 0. # AFAICS WCP offsets are always zero
    return self.header
  def ReadIntEpis(self, i = 0, _fh = None): # Reads all channels of ith episode (stored as one block)
    if not(self.ndata): return np.array('h')
    fh = open(self.fileName, mode = 'rb') if _fh is None else _fh
    fh.seek(self.dataOffset + i * WCP_SECTOR_SIZE_BYTES * (self.header.NBD + self.header.NBA))
    #from scipy.io.numpyio import fread ' now replaced by np.fromfile 
    intdata = np.fromfile(fh, dtype = 'h', count = self.ndata)
    intdata = np.reshape(intdata, (self.nSamp, self.nChan)) # across-channel data is interleaved
    if _fh is None: fh.close()
    return intdata
  def ReadIntData(self, _fileName = None): # output dimensionality: [episode, sample, channel]
    # Cannot be read in one go because data and analysis appear to alternate across episodes
    if _fileName is not None: self.ReadFileInfo(_fileName)
    if not(self.nData):
      return np.array('h')
    fh = open(self.fileName, 'rb')
    self.IntData = np.empty((self.nEpis, self.nSamp, self.nChan), dtype = 'h')
    for i in range(self.nEpis):
      self.IntData[i,:,:] = self.ReadIntEpis(i, fh)
    fh.close()
    return self.IntData 

