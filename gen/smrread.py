# Yet another SMR file reader. NIH. Translated from my own C++ code which was based on Malcolm Lidierth's MATLAB code.

# Gary Bhumbra

import numpy as np
import struct as st

class BlockHeader:
  channelHeader = None
  header = None
  precBlock = 0
  succBlock = 1
  def __init__(self, fh = None, _channelHeader = None):
    if fh is not None and _channelHeader is not None:
      self.readInfo(fh, _channelHeader)
  def readInfo(self, fh = None, _channelHeader = None):
    self.channelHeader = _channelHeader
    if self.channelHeader.blocks is None:
      raise ValueError("Cannot read block headers before reading channel header.")
    self.header = np.zeros((self.channelHeader.blocks, 6), dtype = int)
    if self.channelHeader.firstBlock == -1: return self.header
    fh.seek(self.channelHeader.firstBlock)
    for i in range(self.channelHeader.blocks):
      if i: fh.seek(self.header[i-1, self.succBlock])
      self.header[i, :4] =  st.unpack('4i', fh.read(16))
      self.header[i, 4:6] = st.unpack('2h', fh.read(4))
      if i: self.header[i-1, self.precBlock] = self.header[i, self.precBlock] # realign precBlock to return `currBlock'
    if self.channelHeader.blocks > 1:
      self.header[-1, self.precBlock] = self.header[-2, self.succBlock]
    return self.header

class ChannelHeader:
  delSize = None        # short
  nextDelBlock = None   # int
  firstBlock = None     # int
  lastBlock = None      # int
  blocks = None         # short
  nExtra = None         # short
  preTrig = None        # short
  free0 = None          # short
  phySz = None          # short
  maxData = None        # short
  commentSize = None    # unsigned char
  comment = None        # char[71]
  maxChanTime = None    # int
  lChanDvd = None       # int
  phyChan = None        # short
  titleSize = None      # unsigned char
  title = None          # char[9]
  idealRate = None      # float
  kind = None           # unsigned char
  pad = None            # char
  scale = None          # float
  offset = None         # float
  unitsSize = None      # unsigned char
  units = None          # char[5]
  divideOrInterleave = None # short
  blockHeader = None
  header = None
  baseOffset = 512
  chanDelta = 140
  def __init__(self, fh = None, chan = 0):
    if fh is not None:
      self.readInfo(fh, chan)
  def readInfo(self, fh = None, chan = 0):
    fh.seek(self.baseOffset + self.chanDelta*chan)
    self.delSize = st.unpack('h', fh.read(2))[0]
    self.nextDelBlock = st.unpack('i', fh.read(4))[0]
    self.firstBlock = st.unpack('i', fh.read(4))[0]
    self.lastBlock = st.unpack('i', fh.read(4))[0]
    self.blocks = st.unpack('h', fh.read(2))[0]
    self.nExtra = st.unpack('h', fh.read(2))[0]
    self.preTrig = st.unpack('h', fh.read(2))[0]
    self.free0 = st.unpack('h', fh.read(2))[0]
    self.phySz = st.unpack('h', fh.read(2))[0]
    self.maxData = st.unpack('h', fh.read(2))[0]
    self.commentSize = st.unpack('B', fh.read(1))[0]
    self.comment = fh.read(71)
    self.comment = self.comment[:self.commentSize]
    self.maxChanTime = st.unpack('i', fh.read(4))[0]
    self.lChanDvd = st.unpack('i', fh.read(4))[0]
    self.phyChan = st.unpack('h', fh.read(2))[0]
    self.titleSize = st.unpack('B', fh.read(1))[0]
    self.title = fh.read(9)
    self.title = self.title[:self.titleSize]
    self.idealRate = st.unpack('f', fh.read(4))[0]
    self.kind = st.unpack('B', fh.read(1))[0]
    self.pad = fh.read(1)
    if self.kind in [1, 6]: 
      self.scale = st.unpack('f', fh.read(4))[0]
      self.offset = st.unpack('f', fh.read(4))[0]
      self.unitSize = st.unpack('B', fh.read(1))[0]
      self.units = fh.read(5)
      self.units = self.units[:self.unitSize]
      self.divideOrInterleave = st.unpack('h', fh.read(2))[0]
    self.blockHeader = BlockHeader(fh, self)

class FileHeader:
  systemID = None       # short
  copyright = None      # char[10]
  creator = None        # char[8]
  usPerTime = None      # short
  timePerADC = None     # short
  fileState = None      # short
  firstData = None      # int
  channels = None       # short
  chanSize = None       # short
  extraData = None      # short
  bufferSize = None     # short
  osFormat = None       # short
  maxFTime = None       # int
  dTimeBase = None      # double
  timeDateDetail = None # unsigned char[6]
  timeDateYear = None   # short
  channelHeaders = None
  def __init__(self, fh = None, offset = 0):
    if fh is not None:
      self.readInfo(fh, offset = 0)
  def readInfo(self, fh = None, offset = 0):
    fh.seek(offset)
    self.systemID = st.unpack('h', fh.read(2))[0]
    self.copyright = fh.read(10)
    self.creator = fh.read(8)
    self.usPerTime = st.unpack('h', fh.read(2))[0]
    self.timePerADC = st.unpack('h', fh.read(2))[0]
    self.fileState = st.unpack('h', fh.read(2))[0]
    self.firstData = st.unpack('i', fh.read(4))[0]
    self.channels = st.unpack('h', fh.read(2))[0]
    self.chanSize = st.unpack('h', fh.read(2))[0]
    self.extraData = st.unpack('h', fh.read(2))[0]
    self.bufferSize = st.unpack('h', fh.read(2))[0]
    self.osFormat = st.unpack('h', fh.read(2))[0]
    self.maxFTime = st.unpack('I', fh.read(4))[0]
    self.dTimeBase = st.unpack('d', fh.read(8))[0]
    self.timeDateDetail = st.unpack('6B', fh.read(6))
    self.timeDateYear = st.unpack('h', fh.read(2))[0]
    self.channelHeaders = [[]] * self.channels
    self.blockHeaders = [[]] * self.channels
    for i in range(self.channels):
      self.channelHeaders[i] = ChannelHeader(fh, i)
      self.blockHeaders[i] = self.channelHeaders[i].blockHeader

class SMR:
  fileName = None
  fileHeader = None
  channelHeaders = None
  blockHeaders = None
  headOffset = 20
  ndata = None
  nData = 0
  IntData = None
  Markers = None
  Indices = None
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.ReadFileInfo(_fileName)
  def ReadFileInfo(self, _fileName = None):
    if _fileName == None: return
    self.fileName = _fileName
    fh = open(self.fileName, mode = 'rb')
    self.fileHeader = FileHeader(fh)
    fh.close()
    self.channelHeaders = self.fileHeader.channelHeaders
    self.blockHeaders = self.fileHeader.blockHeaders
    self.nchan  = self.fileHeader.channels
    self.kinds = [chanhead.kind for chanhead in self.channelHeaders]
    self.nData = 0
    self.ndata = [0] * self.nchan
    I = 0
    for i in range(self.nchan):
      if self.blockHeaders[i] is not None:
        I = i
        self.ndata[i] = np.sum(self.blockHeaders[i].header[:,5])
        if self.kinds[i] in [1, 6]:
          self.nData = max(self.nData, self.ndata[i])
    self.nchan = I
    self.kinds = self.kinds[:I]
    self.ndata = self.ndata[:I]
  def ReadIntData(self, _fileName = None, chan = None):
    if _fileName is not None: self.ReadFileInfo(_fileName)
    if not(self.nData):
      return np.array('h')
    if type(chan) is int: chan = [chan]
    if chan is None: chan = range(self.nchan)
    self.intchan = chan[:] # keep a record of channels read
    nchan = len(chan)
    self.IntData = [[]] * nchan
    self.Markers = [[]] * nchan
    self.Indices = [[]] * nchan
    fh = open(self.fileName, 'rb')
    for h in range(nchan):
      c = chan[h]
      head = self.blockHeaders[c].header
      kind = self.kinds[c]
      if kind == 1: # waveform
        self.Indices[h] = np.array([0], dtype = 'i')
        self.Markers[h] = np.array('B')
        self.IntData[h] = np.empty((1, self.ndata[c]), dtype = 'h')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + max(0, head[k,0]))
          d = head[k, 5]
          j = i + d
          self.IntData[h][0, i:j] = st.unpack(str(d)+'h', fh.read(d*2))
          i = j
      elif kind in [2, 3, 4]: # evt-, evt+, evt+/-
        self.Indices[h] = np.empty(self.ndata[c], dtype = 'i')
        self.Markers[h] = np.array('B')
        self.IntData[h] = np.array('h')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + head[k, 0])
          d = head[k, 5]
          j = i + d
          self.Indices[h][i:j] = st.unpack(str(d)+'i', fh.read(d*4))
          i = j
      elif kind == 5: # mark
        self.Indices[h] = np.empty(self.ndata[c], dtype = 'i')
        self.Markers[h] = np.empty( (self.ndata[c], 4), dtype = 'B')
        self.IntData[h] = np.array('h')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + head[k, 0])
          for j in range(head[k, 5]):
            self.Indices[h][i] = st.unpack('i', fh.read(4))[0]
            self.Markers[h][i] = st.unpack('4B', fh.read(4))[0]
            i += 1
      elif kind == 6: # wavemark
        nVal = self.channelHeaders[c].nExtra
        nval = nVal/2
        self.Indices[h] = np.empty(self.ndata[c], dtype = 'i')
        self.Markers[h] = np.empty( (self.ndata[c], 4), dtype = 'B')
        self.IntData[h] = np.empty( (self.ndata[c], nval), dtype = 'h')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + head[k, 0])
          for j in range(head[k, 5]):
            self.Indices[h][i] = st.unpack('i', fh.read(4))[0]
            self.Markers[h][i] = st.unpack('4B', fh.read(4))[0]
            self.IntData[h][i,:] = st.unpack(str(nval)+'h', fh.read(nVal))
            i += 1
      elif kind == 7: # realmark
        nVal = self.channelHeaders[c].nExtra
        nval = nVal/4
        self.Indices[h] = np.empty(self.ndata[c], dtype = 'i')
        self.Markers[h] = np.empty( (self.ndata[c], 4), dtype = 'B')
        self.IntData[h] = np.empty( (self.ndata[c], nval), dtype = 'f')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + head[k, 0])
          for j in range(head[k, 5]):
            self.Indices[h][i] = st.unpack('i', fh.read(4))[0]
            self.Markers[h][i] = st.unpack('4B', fh.read(4))[0]
            self.IntData[h][i,:] = st.unpack(str(nval)+'d', fh.read(nVal))
            i += 1
      elif kind == 8: # realwave
        nVal = self.channelHeaders[c].nExtra
        nval = nVal
        self.Indices[h] = np.empty(self.ndata[c], dtype = 'i')
        self.Markers[h] = np.empty( (self.ndata[c], 4), dtype = 'B')
        self.IntData[h] = np.empty( (self.ndata[c], nval), dtype = 'B')
        i = 0
        for k in range(self.channelHeaders[c].blocks):
          fh.seek(self.headOffset + head[k, 0])
          for j in range(head[k, 5]):
            self.Indices[h][i] = st.unpack('i', fh.read(4))[0]
            self.Markers[h][i] = st.unpack('B', fh.read(4))[0]
            self.IntData[h][i,:] = st.unpack('B', fh.read(nVal))
            i += 1
      else: # dunno
        print("Warning: unknown channel kind: #" + str(kind))
        self.Indices[h] = np.array('i')
        self.Markers[h] = np.array('B')
        self.IntData[h] = np.array('h')
    fh.close()
    return self.IntData

