# ABF File reader (episodic data)
# Nabbed from Harald Hentschke's and Forrest Collman's MATLAB code.
# Gary Bhumbra

import numpy as np
import struct as st

class SectionInfo:  
  def __init__(self, fh = None, offset = None):
    if fh is not None and offset is not None:
      self.readInfo(fh, offset)
  def readInfo(self, fh = None, offset = None):
    fh.seek(offset)
    self.uBlockIndex = st.unpack('I', fh.read(4))[0]
    fh.seek(offset + 4)
    self.uBytes = st.unpack('I', fh.read(4))[0]
    fh.seek(offset + 8)
    self.llNumEntries = st.unpack('q', fh.read(8))[0]

class ADCInfo:
  def __init__(self, fh = None, offset = None):
    if fh is not None and offset is not None:
      self.readInfo(fh, offset)
  def readInfo(self, fh = None, offset = None):
    fh.seek(offset)
    self.nADCNum = st.unpack('h', fh.read(2))[0]
    self.nTelegraphEnable = st.unpack('h', fh.read(2))[0]
    self.nTelegraphInstrument = st.unpack('h', fh.read(2))[0]
    self.fTelegraphAdditGain = st.unpack('f', fh.read(4))[0]
    self.fTelegraphFilter = st.unpack('f', fh.read(4))[0]
    self.fTelegraphMembraneCap = st.unpack('f', fh.read(4))[0]
    self.nTelegraphMode = st.unpack('h', fh.read(2))[0]
    self.fTelegraphAccessResistance = st.unpack('f', fh.read(4))[0]
    self.nADCPtoLChannelMap = st.unpack('h', fh.read(2))[0]
    self.nADCSamplingSeq = st.unpack('h', fh.read(2))[0]
    self.fADCProgrammableGain = st.unpack('f', fh.read(4))[0]
    self.fADCDisplayAmplification = st.unpack('f', fh.read(4))[0]
    self.fDisplayOffset = st.unpack('f', fh.read(4))[0]
    self.fInstrumentScaleFactor = st.unpack('f', fh.read(4))[0]
    self.fInstrumentOffset = st.unpack('f', fh.read(4))[0]
    self.fSignalGain = st.unpack('f', fh.read(4))[0]
    self.fSignalOffset = st.unpack('f', fh.read(4))[0]
    self.fSignallLowpassFilter = st.unpack('f', fh.read(4))[0]
    self.fSignalHighpassFilter = st.unpack('f', fh.read(4))[0]
    self.nLowpassFilterType = st.unpack('b', fh.read(1))[0]
    self.nHighpassFilterType = st.unpack('b', fh.read(1))[0]
    self.fPostProcessLowpassFilter = st.unpack('f', fh.read(4))[0]
    self.nPostProcessLowpassFilterType = st.unpack('b', fh.read(1))[0]
    self.bEnabledDuringPN = st.unpack('?', fh.read(1))[0] # meh!
    self.nStatsChannelPolarity = st.unpack('h', fh.read(2))[0]
    self.lADCChannelNameIndex = st.unpack('I', fh.read(4))[0]
    self.lADCUnitsIndex = st.unpack('I', fh.read(4))[0]

class ProtocolInfo:
  def __init__(self, fh = None, offset = None):
    if fh is not None and offset is not None:
      self.readInfo(fh, offset)
  def readInfo(self, fh = None, offset = None):
    fh.seek(offset)
    self.nOperationMode = st.unpack('h', fh.read(2))[0]
    self.fADCSequenceInterval = st.unpack('f', fh.read(4))[0]
    self.bEnableFileCompression = st.unpack('?', fh.read(1))[0] #meh!
    self.sUnused1 = st.unpack('b'*3, fh.read(3))[0]
    self.uFileCompressionRatio = st.unpack('I', fh.read(4))[0]
    self.fSynchTimeUnit = st.unpack('f', fh.read(4))[0]
    self.fSecondsPerRun = st.unpack('f', fh.read(4))[0]
    self.lNumSamplesPerEpisode = st.unpack('i', fh.read(4))[0]
    self.lPreTriggerSamples = st.unpack('i', fh.read(4))[0]
    self.lEpisodesPerRun = st.unpack('i', fh.read(4))[0]
    self.lRunsPerTrial = st.unpack('i', fh.read(4))[0]
    self.lNumberOfTrials = st.unpack('i', fh.read(4))[0]
    self.nAveragingMode = st.unpack('h', fh.read(2))[0]
    self.nUndoRuncount = st.unpack('h', fh.read(2))[0]
    self.nFirstEpisodeInRun = st.unpack('h', fh.read(2))[0]
    self.fTriggerThreshold = st.unpack('f', fh.read(4))[0]
    self.nTriggerSource = st.unpack('h', fh.read(2))[0]
    self.nTriggerAction = st.unpack('h', fh.read(2))[0]
    self.nTriggerPolarity = st.unpack('h', fh.read(2))[0]
    self.fScopeOutputInterval = st.unpack('f', fh.read(4))[0]
    self.fEpisodeStartToStart = st.unpack('f', fh.read(4))[0]
    self.fRunStartToStart = st.unpack('f', fh.read(4))[0]
    self.lAverageCount = st.unpack('i', fh.read(4))[0]
    self.fTrialStartToStart = st.unpack('f', fh.read(4))[0]
    self.nAutoTriggerStrategy = st.unpack('h', fh.read(2))[0]
    self.fFirstRunDelayS = st.unpack('f', fh.read(4))[0]
    self.nChannelStatsStrategy = st.unpack('h', fh.read(2))[0]
    self.lSamplesPerTrace = st.unpack('i', fh.read(4))[0]
    self.lStartDisplayNum = st.unpack('i', fh.read(4))[0]
    self.lFinishDisplayNum = st.unpack('i', fh.read(4))[0]
    self.nShowPNRawData = st.unpack('h', fh.read(2))[0]
    self.fStatisticsPeriod = st.unpack('f', fh.read(4))[0]
    self.lStatisticsMeasurements = st.unpack('i', fh.read(4))[0]
    self.nStatisticsSaveStrategy = st.unpack('h', fh.read(2))[0]
    self.fADCRange = st.unpack('f', fh.read(4))[0]
    self.fDACRange = st.unpack('f', fh.read(4))[0]
    self.lADCResolution = st.unpack('i', fh.read(4))[0]
    self.lDACResolution = st.unpack('i', fh.read(4))[0]
    self.nExperimentType = st.unpack('h', fh.read(2))[0]
    self.nManualInfoStrategy = st.unpack('h', fh.read(2))[0]
    self.nCommentsEnable = st.unpack('h', fh.read(2))[0]
    self.lFileCommentIndex = st.unpack('i', fh.read(4))[0]
    self.nAutoAnalyseEnable = st.unpack('h', fh.read(2))[0]
    self.nSignalType = st.unpack('h', fh.read(2))[0]
    self.nDigitalEnable = st.unpack('h', fh.read(2))[0]
    self.nActiveDACChannel = st.unpack('h', fh.read(2))[0]
    self.nDigitalHolding = st.unpack('h', fh.read(2))[0]
    self.nDigitalInterEpisode = st.unpack('h', fh.read(2))[0]
    self.nDigitalDACChannel = st.unpack('h', fh.read(2))[0]
    self.nDigitalTrainActiveLogic = st.unpack('h', fh.read(2))[0]
    self.nStatsEnable = st.unpack('h', fh.read(2))[0]
    self.nStaticsClearStrategy = st.unpack('h', fh.read(2))[0]
    self.nLevelHysteris = st.unpack('h', fh.read(2))[0]
    self.lTimeHysteresis = st.unpack('i', fh.read(4))[0]
    self.nAllowExternalTags = st.unpack('h', fh.read(2))[0]
    self.nAverageAlgorithm = st.unpack('h', fh.read(2))[0]
    self.lTimeHysteresis = st.unpack('f', fh.read(4))[0]
    self.nUndoPromptStategy = st.unpack('h', fh.read(2))[0]
    self.nTrialTriggerSource = st.unpack('h', fh.read(2))[0]
    self.nStatisticsDisplayStrategy = st.unpack('h', fh.read(2))[0]
    self.nExternalTagType = st.unpack('h', fh.read(2))[0]
    self.nScopeTriggerOut = st.unpack('h', fh.read(2))[0]
    self.nLTPType = st.unpack('h', fh.read(2))[0]
    self.nAlternateDACOutputState = st.unpack('h', fh.read(2))[0]
    self.nAlternateDigitalOutputState = st.unpack('h', fh.read(2))[0]
    self.fCellID = st.unpack('f', fh.read(4))[0]
    self.nDigitizerADCs = st.unpack('h', fh.read(2))[0]
    self.nDigitizerDACs = st.unpack('h', fh.read(2))[0]
    self.nDigitizerTotalDigitalOuts = st.unpack('h', fh.read(2))[0]
    self.nDigitizerSynchDigitalOuts = st.unpack('h', fh.read(2))[0]
    self.nDigitizerType = st.unpack('h', fh.read(2))[0]

class ABFile:
  fileName = None
  ABFVersion = None
  nData = 0
  def __init__(self, _fileName = None):
    if _fileName == None: return
    self.ReadFileInfo(_fileName)
  def ReadFileInfo(self, _fileName = None):
    if _fileName == None: return
    self.fileName = _fileName
    self.ReadFileHeader()
    self.ReadADCChannelInfo()  
  def ReadFileSignature(self):
    fh = open(self.fileName, mode = 'rb')
    fh.seek(0); self.uFileSignature = st.unpack('I', fh.read(4))[0]
    fh.close()
    self.ABFVersion = None 
    if self.uFileSignature == 541475393:
      self.ABFVersion = 1
    elif self.uFileSignature == 843465281:
      self.ABFVersion = 2
    return self.uFileSignature
  def ReadFileHeader(self, blockSize = 512):
    if self.fileName is None: return
    if self.ABFVersion is None: self.ReadFileSignature()
    if self.ABFVersion == 1:
      fh = open(self.fileName, mode = 'rb')
      fh.seek(0); self.uFileSignature = st.unpack('I', fh.read(4))[0]
      fh.seek(4); self.fFileVersionNumber = st.unpack('f', fh.read(4))[0]
      fh.seek(8); self.nOperationMode = st.unpack('h', fh.read(2))[0]
      fh.seek(10); self.lActualAcqLength = st.unpack('I', fh.read(4))[0]
      fh.seek(14); self.nNumPointsIgnored  = st.unpack('h', fh.read(2))[0]
      fh.seek(16); self.lActualEpisodes = st.unpack('I', fh.read(4))[0]
      fh.seek(24); self.lFileStartTime = st.unpack('I', fh.read(4))[0]
      fh.seek(40); self.lDataSectionPtr = st.unpack('I', fh.read(4))[0]
      fh.seek(92); self.lSynchArrayPtr = st.unpack('I', fh.read(4))[0]
      fh.seek(96); self.lSynchArraySize = st.unpack('I', fh.read(4))[0]
      fh.seek(100); self.nDataFormat = st.unpack('h', fh.read(2))[0]
      fh.seek(120); self.nADCNumChannels = st.unpack('h', fh.read(2))[0]
      fh.seek(122); self.fADCSampleInterval = st.unpack('f', fh.read(4))[0]
      fh.seek(130); self.fSynchTimeUnit = st.unpack('f', fh.read(4))[0]
      fh.seek(138); self.lNumSamplesPerEpisode = st.unpack('I', fh.read(4))[0]
      fh.seek(142); self.lPreTriggerSamples = st.unpack('I', fh.read(4))[0]
      fh.seek(146); self.lEpisodesPerRun = st.unpack('I', fh.read(4))[0]
      fh.seek(244); self.fADCRange = st.unpack('f', fh.read(4))[0]
      fh.seek(252); self.lADCResolution = st.unpack('I', fh.read(4))[0]
      fh.seek(366); self.nFileStartMillisecs = st.unpack('h', fh.read(2))[0]       
      fh.seek(378); self.nADCPtoLChannelMap = st.unpack('h '*16, fh.read(2*16)) 
      fh.seek(410); self.nADCSamplingSeq = st.unpack('h '*16, fh.read(2*16)) 
      seekOffset = 442
      self.sADCChannelName = [[]] * 16
      self.sADCChannelUnits = ['units'] * 16
      for i in range(16):
        fh.seek(seekOffset)
        self.sADCChannelName[i] = fh.read(10)
        seekOffset += 10
      fh.seek(730); self.fADCProgrammableGain = st.unpack('f'*16, fh.read(4*16))
      fh.seek(922); self.fInstrumentScaleFactor = st.unpack('f'*16, fh.read(4*16))
      fh.seek(986); self.fInstrumentOffset = st.unpack('f'*16, fh.read(4*16))
      fh.seek(1050); self.fSignalGain = st.unpack('f'*16, fh.read(4*16))
      fh.seek(1114); self.fSignalOffset = st.unpack('f'*16, fh.read(4*16))
      fh.seek(4512); self.nTelegraphEnable = st.unpack('h'*16, fh.read(2*16))
      fh.seek(4576); self.fTelegraphAdditGain = st.unpack('f'*16, fh.read(4*16))
      fh.close()
      self.fFileStartTime = self.nFileStartMillisecs * 0.001 
      self.recChNames = self.sADCChannelName
      self.recChUnits = self.sADCChannelUnits
    elif self.ABFVersion == 2:
      self.iFileVersionNumber = [[]] * 4
      fh = open(self.fileName, mode = 'rb')
      fh.seek(0); self.uFileSignature = st.unpack('I', fh.read(4))[0]
      fh.seek(4); self.iFileVersionNumber[0] = st.unpack('B', fh.read(1))[0]
      fh.seek(5); self.iFileVersionNumber[1] = st.unpack('B', fh.read(1))[0]
      fh.seek(6); self.iFileVersionNumber[2] = st.unpack('B', fh.read(1))[0]
      fh.seek(7); self.iFileVersionNumber[3] = st.unpack('B', fh.read(1))[0]
      fh.seek(8); self.uFileInfoSize = st.unpack('I', fh.read(4))[0]
      fh.seek(12); self.lActualEpisodes = st.unpack('I', fh.read(4))[0]
      fh.seek(16); self.uFileStartDate  = st.unpack('I', fh.read(4))[0]
      fh.seek(20); self.uFileStartTimeMS = st.unpack('I', fh.read(4))[0]
      fh.seek(24); self.uStopwatchTime = st.unpack('I', fh.read(4))[0]
      fh.seek(28); self.nFileType = st.unpack('h', fh.read(2))[0]
      fh.seek(30); self.nDataFormat = st.unpack('h', fh.read(2))[0]
      fh.seek(32); self.nSimultaneousScan = st.unpack('h', fh.read(2))[0]
      fh.seek(34); self.nCRCEnable = st.unpack('h', fh.read(2))[0]
      fh.seek(36); self.uFileCRC = st.unpack('I', fh.read(4))[0]
      fh.seek(40); self.FileGUID = st.unpack('I', fh.read(4))[0]
      fh.seek(56); self.uCreatorVersion = st.unpack('I', fh.read(4))[0]
      fh.seek(60); self.uCreatorNameIndex = st.unpack('I', fh.read(4))[0]
      fh.seek(64); self.uModifierVersion = st.unpack('I', fh.read(4))[0]
      fh.seek(68); self.uModifierNameIndex = st.unpack('I', fh.read(4))[0]
      fh.seek(72); self.uProtocolPathIndex = st.unpack('I', fh.read(4))[0]
      fh.close()      
      self.fFileVersionNumber = self.iFileVersionNumber[3] + self.iFileVersionNumber[2] * .1
      self.fFileVersionNumber += self.iFileVersionNumber[1] * .01 +  self.iFileVersionNumber[0] * .001  
      self.fFileStartTime = self.uFileStartTimeMS * 0.001
      self.ReadSections()
      self.ReadStrings(blockSize)
      self.ReadADCInfo(blockSize)
      self.ReadProtocol(blockSize)
    self.dataTypeSize = 2 if self.nDataFormat else 4
    self.headOffset = self.lDataSectionPtr * blockSize + self.nNumPointsIgnored * self.dataTypeSize
    self.totalSamplingInterval = self.fADCSampleInterval * self.nADCNumChannels
    self.samplint = float(self.fADCSampleInterval) * float(self.nADCNumChannels) * 1e-6
    self.nChannels = int(self.nADCNumChannels)
    self.nEpisodes = max(1, int(self.lActualEpisodes))
    self.nSamples = self.lActualAcqLength // int(self.nEpisodes * self.nChannels)
    self.nData = int(self.nSamples * self.nEpisodes * self.nChannels)
  def ReadSections(self, _offset = 76, _delta = 16):
    offset = _offset
    delta = _delta
    fh = open(self.fileName, mode = 'rb')
    self.ProtocolSection = SectionInfo(fh, offset); offset += delta
    self.ADCSection = SectionInfo(fh, offset); offset += delta
    self.DACSection = SectionInfo(fh, offset); offset += delta
    self.EpochSection = SectionInfo(fh, offset); offset += delta
    self.ADCPerDACSection = SectionInfo(fh, offset); offset += delta
    self.EpochPerDACSection = SectionInfo(fh, offset); offset += delta
    self.UserListSection = SectionInfo(fh, offset); offset += delta
    self.StatsRegionSection = SectionInfo(fh, offset); offset += delta
    self.MathSection = SectionInfo(fh, offset); offset += delta
    self.StringsSection = SectionInfo(fh, offset); offset += delta
    self.DataSection = SectionInfo(fh, offset); offset += delta
    self.TagSection = SectionInfo(fh, offset); offset += delta
    self.ScopeSection = SectionInfo(fh, offset); offset += delta
    self.DeltaSection = SectionInfo(fh, offset); offset += delta
    self.VoiceTagSection = SectionInfo(fh, offset); offset += delta
    self.SynchArraySection = SectionInfo(fh, offset); offset += delta
    self.AnnotationSection = SectionInfo(fh, offset); offset += delta
    self.StatsSection = SectionInfo(fh, offset); offset += delta
    fh.close()
  def ReadStrings(self, blockSize = 512):
    fh = open(self.fileName, mode = 'rb')
    fh.seek(self.StringsSection.uBlockIndex * blockSize)
    stringssection = fh.read(self.StringsSection.uBytes).decode(errors='replace')
    fh.close()
    i = stringssection.lower().find('clampex')
    if i < 0: i = stringssection.lower().find('axoscope')
    stringssection = stringssection[i:]
    i = stringssection.find(chr(0))
    self.Strings = []
    while i>=0:
      self.Strings.append(stringssection[:i])
      stringssection = stringssection[(i+1):]
      i = stringssection.find(chr(0))
  def ReadADCInfo(self, blockSize = 512):
    self.ADCSections = [[]] * self.ADCSection.llNumEntries
    fh = open(self.fileName, mode = 'rb') 
    for i in range(self.ADCSection.llNumEntries):
      self.ADCSections[i] = ADCInfo(fh, int(self.ADCSection.uBlockIndex*blockSize)+\
                                        int(self.ADCSection.uBytes*i))
    fh.close()
    j = 0
    for i in range(self.ADCSection.llNumEntries):
      j = max(j, self.ADCSections[i].nADCNum)
    j += 1  
    self.recChNames = [''] * j      
    self.recChUnits = [''] * j
    self.nADCSamplingSeq = [[]] * j
    self.nTelegraphEnable = [[]] * j
    self.fTelegraphAdditGain = [[]] * j
    self.fInstrumentScaleFactor = [[]] * j
    self.fSignalGain = [[]] * j
    self.fADCProgrammableGain = [[]] * j
    self.fInstrumentOffset = [[]] * j
    self.fSignalOffset = [[]] * j
    for i in range(self.ADCSection.llNumEntries):
      self.recChNames[i] = self.Strings[self.ADCSections[i].lADCChannelNameIndex - 1] 
      self.recChUnits[i] = self.Strings[self.ADCSections[i].lADCUnitsIndex - 1]
      j = self.ADCSections[i].nADCNum
      self.nADCSamplingSeq[i] = j
      self.nTelegraphEnable[j] = self.ADCSections[i].nTelegraphEnable;
      self.fTelegraphAdditGain[j] = self.ADCSections[i].fTelegraphAdditGain;
      self.fInstrumentScaleFactor[j] = self.ADCSections[i].fInstrumentScaleFactor;
      self.fSignalGain[j] = self.ADCSections[i].fSignalGain;
      self.fADCProgrammableGain[j] = self.ADCSections[i].fADCProgrammableGain;
      self.fInstrumentOffset[j] = self.ADCSections[i].fInstrumentOffset;
      self.fSignalOffset[j] = self.ADCSections[i].fSignalOffset;
  def ReadProtocol(self, blockSize = 512):    
    fh = open(self.fileName, mode = 'rb') 
    self.Protocol = ProtocolInfo(fh, self.ProtocolSection.uBlockIndex*blockSize)
    self.nOperationMode = self.Protocol.nOperationMode
    self.fSynchTimeUnit = self.Protocol.fSynchTimeUnit
    self.lPreTriggerSamples = self.Protocol.lPreTriggerSamples
    self.nADCNumChannels = self.ADCSection.llNumEntries
    self.lActualAcqLength = self.DataSection.llNumEntries
    self.lDataSectionPtr = self.DataSection.uBlockIndex
    self.nNumPointsIgnored = 0
    self.fADCSampleInterval = float(self.Protocol.fADCSequenceInterval)/float(self.nADCNumChannels)
    self.fADCRange = self.Protocol.fADCRange
    self.lADCResolution = self.Protocol.lADCResolution
    self.lSynchArrayPtr = self.SynchArraySection.uBlockIndex
    self.lSynchArraySize = self.SynchArraySection.llNumEntries
  def ReadADCChannelInfo(self):
    self.name = [''] * self.nChannels
    self.units = [''] * self.nChannels
    self.index = np.empty(self.nChannels, dtype = int)
    self.addGain = np.ones(self.nChannels, dtype = float)
    self.index = np.arange(self.nChannels)
    self.name = self.recChNames[:self.nChannels]
    self.units = self.recChUnits[:self.nChannels]
    if self.fFileVersionNumber > 1.649:
      for i in range(self.nChannels):
        ch = self.nADCSamplingSeq[i]
        if self.nTelegraphEnable[ch]:
          self.addGain[i] = self.fTelegraphAdditGain[ch]
        else:
          self.addGain[i] = 1
    ADCGain = self.fADCRange / self.lADCResolution
    self.gain = np.ones(self.nChannels, dtype = float)
    self.offset = np.zeros(self.nChannels, dtype = float)
    for i in range(self.nChannels):
      ch = self.nADCSamplingSeq[i]
      rGainCh = self.fInstrumentScaleFactor[ch] * self.fSignalGain[ch] * self.fADCProgrammableGain[ch] * self.addGain[i]
      self.gain[i] = ADCGain / rGainCh
      self.offset[i] = self.fInstrumentOffset[ch] * self.fSignalOffset[ch]
  def ReadIntData(self):
    #from scipy.io.numpyio import fread ' now replaced by np.fromfile 
    if not(self.nData):
      return np.array('h')
    fh = open(self.fileName, 'rb')
    fh.seek(self.headOffset)
    self.IntData = np.fromfile(fh, dtype = 'h', count = self.nData)
    fh.close()
    return self.IntData.reshape( (self.nEpisodes, self.nSamples, self.nChannels) )
