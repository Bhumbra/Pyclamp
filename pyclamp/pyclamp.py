DEVMODE = True # Set to `False' for excluding the corrupting effects of bug-ridden coding experiments

import sys
import os
CWDIR = os.getcwd()
MANDIR = 'man'
MANPREF = "pyclamp"
MANSUFF = ".html"
MANHELP = [""] 
DEFAULT_TDF_EXTENSION = ".tdf"

import gc
import webbrowser
from PyQt6 import QtGui
import numpy as np
import pyclamp.dsp.sifunc as sifunc
import pyclamp.dsp.wffunc as wffunc
import pyclamp.dsp.fpanal as fpanal
import pyclamp.dsp.wfanal as wfanal
import pyclamp.dsp.iofunc as iofunc
import pyclamp.dsp.channel as channel

from pyclamp.qnp.qmod import Qmodl
import pyclamp.gui.lbwgui as lbw
import pyclamp.dsp.multio as mio
from pyclamp.gui.pyplot import pywav
import pyclamp.gui.pydisc as pydisc
from pyclamp.gui.pysumm import pysumm

from pyclamp.gui.pyqtplot import *
from pyclamp.gui.pgb import *
from pyclamp.dsp.dtypes import *
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.strfunc import *
from pyclamp.dsp.tdf import *
from pyclamp.dsp.tpfunc import *

def main():                  # OS entry point 
  return PyClamp()  
  
class PyClamp:          # front-end controller
  Form = None
  Area = None
  defxy = [800, 600]
  pc = None
  qa = None
  def __init__(self, path = None):
    self.child = None
    self.Child = None
    self.children = []
    self.Children = []
    self.App = lbw.LBWidget(None, None, None, 'app')
    self.Form = lbw.LBWidget(None, None, None, 'mainform', "Pyclamp")
    self.Area = lbw.LBWidget(None, None, None, 'childarea')
    self.Form.setChild(self.Area)
    '''
    self.Box = lbw.LBWidget(self.Form, None, 1)
    self.BG0 = lbw.BGWidgets(self.Form, 'Open', 0, ["Data file(s)", "Analysis file", "Results file"], 'radiobutton', 0) 
    self.BBx = lbw.BWidgets(self.Form, 0, None, ["Button", "Button", "Button"], ["Help", "OK", "Exit"]) 
    self.BBx.Widgets[0].connect("btndown", self.inithelp)
    self.BBx.Widgets[1].connect("btndown", self.initialise)
    self.BBx.Widgets[2].connect("btndown", self.exitfunc)            
    self.Box.add(self.BG0)
    self.Box.add(self.BBx)
    self.Form.setChild(self.Box)
    '''
    dataAction = QtGui.QAction(QtGui.QIcon('open.png'), 'Open &Data File', self.Form.FWidget)        
    dataAction.setShortcut('Ctrl+D')
    dataAction.setStatusTip('Open Data file')
    dataAction.triggered.connect(self.openData)
    analAction = QtGui.QAction(QtGui.QIcon('open.png'), 'Open &Analysis File', self.Form.FWidget)        
    analAction.setShortcut('Ctrl+A')
    analAction.setStatusTip('Open Analysis file')
    analAction.triggered.connect(self.openAnal)
    resuAction = QtGui.QAction(QtGui.QIcon('open.png'), 'Open &Results File', self.Form.FWidget)        
    resuAction.setShortcut('Ctrl+R')
    resuAction.setStatusTip('Open Results file')
    resuAction.triggered.connect(self.openResu)
    exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self.Form.FWidget)        
    exitAction.setShortcut('Ctrl+Q')
    exitAction.setStatusTip('Exit application')
    exitAction.triggered.connect(self.exitfunc)
    self.Form.FWidget.statusBar()
    menubar = self.Form.FWidget.menuBar()
    fileMenu = menubar.addMenu('&File')
    fileMenu.addAction(dataAction)
    fileMenu.addAction(analAction)
    fileMenu.addAction(resuAction)
    fileMenu.addAction(exitAction)
    self.Form.FWidget.resize(self.defxy[0], self.defxy[1])
    self.Form.FWidget.show()  
  def inithelp(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[0]+MANSUFF)
  def openData(self, ev = None):
    #self.Child = self.Area.addChild('Data File')
    self.Child = lbw.LBWidget(None, None, None, 'mainform', "Data File")
    self.Children.append(self.Child)
    self.child = Pyclamp(self.Child)
    self.children.append(self.child)
    self.child.readData()
  def openAnal(self, ev = None):
    self.Child = self.Area.addChild('Analysis File')
    self.Children.append(self.Child)
    self.child = Qmodl()
    self.children.append(self.child)
    self.child.iniForm(self.Child)
    self.child.openFile()
  def openResu(self, ev = None):
    self.Child = self.Area.addChild('Analysis File')
    self.Children.append(self.Child)
    self.child = Qmodl()
    self.children.append(self.child)
    self.child.iniForm(self.Child)
    self.child.Archive()
  def exitfunc(self, event = None):
    print("User exited.")
    for _Child in self.Children: _Child.close()
    if self.Form is not None: self.Form.FWidget.close()
    if self.App is None: return False
    self.App.Widget.quit()
    return True 

def mpmfilio(q):
  q.put(mio.mfilio())

def mpmfilioGUI(q):
  q.put(mio.mfilioGUI())

class pyclamp:              # back-end
  _Data = None
  mfi = None
  P = None
  active = None
  extracted = None
  selected = None
  comment = None
  ccl = None
  def __init__(self):
    self.initialise()
  def initialise(self):
    self.fmap = {'setDataFiles':self.setDataFiles,
                 'setChannels':self.setChannels,
                 'readIntData':self.readIntData,
                 'setSelection':self.setSelection,
                 'setClampConfig':self.setClampConfig,
                 'time2Index':self.time2Index,
                 'range2Indices':self.range2Indices,
                 'filter':self.filter,
                 'quiesce':self.quiesce,
                 'trim':self.trim,
                 'align':self.align,
                 'trigger':self.trigger,
                 'baseline':self.baseline,
                 'homogenise':self.homogenise,
                 'offset':self.offset,
                 'extract':self.extract,
                 'updatetable':self.updatetable,
                 'writetable':self.writetable,
                 'setActive':self.setActive,
                 'expfit':self.expfit,
                 'psa':self.psa,
                 'mwd':self.mwd,
                 'triggers':self.triggers,
                 'spikeShape':self.spikeShape,
                 'iStep':self.iStep,
                 'vStep':self.vStep,
                 'setLabel':self.setLabel,
                 'setComment':self.setComment}
    self.table = listtable()
    self.log = []
    self.Data = []
    self.nchan = 0
    self.nepis = 0
    self.nsamp = 0
    self.Results = []
    self.Labels = []
    self.IntData = []
    self.nepis = None
    self.SiGnOf = []
    self.Selection = None
    self.Range = None
    self.setsellog = None
    self.trig = None
    self.trigID = None
    self.dataType = None
    self.writeDisabled = False
    self.setDataDir()
    self.setDataFiles()
    self.setChannels()
  def appendLog(self, _log, issetsel = False):
    if not(issetsel) or self.setsellog is None:
      self.log.append(_log)
      if issetsel and self.setsellog is None:
        self.setsellog = len(self.log)
      return
    _log = self.log[:]
    log_ = min(self.setsellog, len(_log))
    self.log = _log[:log_]
  def printLog(self, event = None):
    for _log in self.log:
      print(_log)
  def runstr(self, _str):
    if type(_str) is int: _str = self.log[_str]
    if type(_str) is str: _str = [_str]
    for str_ in _str:
      func, args = parseexpstr(str_)
      self.fmap[func](*args)
  def setDataDir(self, dirn = None):
    self.DataDir = dirn
  def setDataFiles(self, filn = []):
    self.DataFiles = filn 
    self.NumDataFiles = nLen(filn)
    if self.NumDataFiles:
      self.appendLog("setDataFiles(" + str(filn) + ")")
  def setChannels(self, _channels = None, _signof = None):
    self.ChanInfo = []
    self.Channels = _channels
    self.nChan = nLen(_channels)    
    if _channels is None: return
    if _signof is None: # we must conclude that this info is contained in self.Channels
      self.ChanInfo = self.Channels[:]
      self.Channels = []
      self.SiGnOf = []
      for chan in self.ChanInfo:
        self.Channels.append(chan.index)
        self.SiGnOf.append([chan.samplint, chan.gain, chan.offset])
    else:    
      self.SiGnOf = _signof 
    self.appendLog("setChannels(" + str(self.Channels) + ", " + str(self.SiGnOf) + ")")  
  def readIntData(self):
    if self.DataDir is None or self.NumDataFiles == 0: return
    if self.nChan == 0 or self.SiGnOf is None: return
    self.mfi = mio.mfilio()
    self.mfi.readData(self.DataFiles, self.Channels, self.DataDir)
    self.IntData = self.mfi.Data
    self.Onsets = np.copy(self.mfi.Onsets)
    self.nSamp = 0
    if len(self.IntData):
      if len(self.IntData[0]):
        self.Selection = [[]] * len(self.IntData[0])
        for j in range(len(self.IntData[0])):
          self.Selection[j] = -len(self.IntData[0][j])
        if len(self.IntData[0][0]):
          self.nSamp = self.IntData[0][0][0].shape[0]
          if len(self.IntData[0][0][0]):
            self.dataType = type(self.IntData[0][0][0][0])
    self.nEpis = 0
    for intdata in self.IntData[0]: self.nEpis += len(intdata)    
    self.appendLog("readIntData()")
  def setSelection(self, _selection = None, _range = None):
    if _selection != None: self.Selection = _selection
    if _range != None: self.Range = self.Range = _range
    if self.Selection is None: return
    if self.Range is None: self.Range = [-np.inf, np.inf]
    if self._Data is None: self._Data = [None] * self.nChan
    selection = sifunc.list2uint(self.Selection)
    self.nchan, self.nepis, self.nsamp = self.nChan, self.nEpis, self.nSamp
    self.active = None # forces initialisiation of self.active
    if len(self.ChanInfo):
      self.DataInfo = self.ChanInfo[:]
    elif self.nchan:
      self.DataInfo = [None]*self.nchan
      for i in range(self.nchan):
        self.DataInfo[i] = channel.chWave()
        self.DataInfo[i].samplint = self.SiGnOf[i][0]
        self.DataInfo[i].gain = self.SiGnOf[i][1]
        self.DataInfo[i].offset = self.SiGnOf[i][2]
    self.setActive([True, True, False], self.nepis, True)
    self.DataType = float if self.dataType is None else self.dataType
    self.Data = [[]] * self.nchan
    for i in range(self.nchan):
      if self._Data[i] is None:
        self.Data[i] = np.empty( (self.nepis, self.nsamp), dtype = self.DataType)
        counter = 0
        for j in range(len(selection)):
          for k in range(len(selection[j])):
            self.Data[i][counter] = self.IntData[i][j][k]
            counter += 1
        self._Data[i] = np.copy(self.Data[i])
      else:
        self.Data[i] = np.copy(self._Data[i])
    if self.mfi is not None:
      self.dirn, self.stpf  = self.mfi.dirn, self.mfi.stpf # carpe diem
      self.mfi.clear()
      del self.mfi
      self.mfi = None
    if self.P is not None:
      self.P.terminate()
      self.P.join()
      del self.P
      self.P = None
    gc.collect()
    self.onsets = np.copy(self.Onsets)
    if elType(self.onsets) is float and len(self.SiGnOf):
      self.onsets = np.array(self.onsets / float(self.SiGnOf[0][0]), dtype = int)
    self.epiID = np.arange(self.nepis, dtype = int)
    self.appendLog("setSelection(" + str(self.Selection) + ")", True)      
  def setClampConfig(self, _ivClamp = 0, _cpConfig = 0):
    self.ivClamp = _ivClamp
    self.cpConfig = _cpConfig
    if self.ivClamp:
      self.pstext = 'PSC'
      self.psText = 'Post-Synaptic Current'    
    else:
      self.pstext = 'PSP'
      self.psText = 'Post-Synaptic Potential'
    self.ccl = len(self.log)
    self.appendLog("setClampConfig(" + str(self.ivClamp) + ", " + str(self.cpConfig) + ")")
  def time2Index(self, t, sigID = 0, _overlay = True):
    i = t
    if not(type(i) is int): i = int(i/self.SiGnOf[sigID][0])
    self.nsam = len(self.Data[sigID][0])
    mini, maxi = 0, self.nsam
    if not(_overlay): 
      mini, maxi = self.onsets[0], self.onsets[-1]+self.nsam
    i = max(mini, min(maxi, i))
    return i
  def range2Indices(self, _timeRange = None, sigID = 0, _overlay = True):
    timeRange = _timeRange
    if timeRange is None: 
      timeRange = [0, self.Data[sigID].shape[1]] if _overlay else [self.onset[0], self.onset[-1]+self.nsam]
    timeRange[0] = self.time2Index(timeRange[0], sigID, _overlay)
    timeRange[1] = self.time2Index(timeRange[1], sigID, _overlay)
    if timeRange[1] < timeRange[0]: timeRange = [timeRange[1], timeRange[0]]
    return timeRange
  def ADC2int(self, x, sigID = 0):
    if not(isarray(x)):
      return self.DataType( (x - self.SiGnOf[sigID][2]) / (self.SiGnOf[sigID][1]))
    else:
      n = len(x)
      X = [[]] * n
      for i in range(n):
        X[i] = self.ADC2int(x[i], sigID)
      return X
  def trim(self, sigID = 0, _timeRange = None, _overlay = True, notlog = False):
    timeRange = self.range2Indices(_timeRange, sigID, _overlay)
    if _overlay:
      for i in range(len(self.Data)): # this method assumes identical samplint for all channels
        self.Data[i] = self.Data[i][:, timeRange[0]:timeRange[1]]  
      self.onsets += timeRange[0]     # no need to change self.active
    else:
      ilo = np.nonzero(self.onsets < timeRange[0])[0]
      ihi = np.nonzero(self.onsets > timeRange[1])[0]
      ilo = ilo[-1] if len(ilo) else 0
      ihi = ihi[ 0] if len(ihi) else self.nepis
      I = np.arange(self.nepis, dtype = int)
      boolRange = np.logical_and(I >= ilo, I < ihi)
      indRange = np.nonzero(boolRange)[0]
      for i in range(len(self.Data)): # this method assumes identical samplint for all channels
        self.Data[i] = self.Data[i][indRange, :]  
      self.onsets = self.onsets[indRange]
      self.epiID = self.epiID[indRange]
      _active0 = self.active[0][indRange]
      _active1 = self.active[1][indRange]
      _active2 = self.active[2][indRange]
      self.setActive([_active0, _active1, _active2], len(indRange), True)
    if self.trig is not None:
      i = np.nonzero(np.logical_and(self.trig[:,1] >= timeRange[0], self.trig[:,1] < timeRange[1]))[0]
      if len(i):
        self.trig = self.trig[i, :]
        self.trig[:, 1] -= timeRange[0]
      else:
        self.trig = None
        self.trigID = None
    if not(notlog):
      self.appendLog("trim(" + str(sigID) + ", " + str(timeRange) + ", " + str(_overlay) + ")")    
  def filter(self, sigID = 0, mode = None, freq = None, nlin = 0., _pgb = None):
    if mode is None or freq is None: return
    if isnum(freq): freq = [freq, freq]
    if mode:
      self.Data[sigID] = wffunc.hpfilter(self.Data[sigID], 1./ self.SiGnOf[sigID][0], freq, nlin, _pgb)
    else:
      self.Data[sigID] = wffunc.lpfilter(self.Data[sigID], 1./ self.SiGnOf[sigID][0], freq, nlin, _pgb)
    self.appendLog("filter(" + str(sigID) + ", " + str(mode) + ", " + str(freq) + ", " + str(nlin) + ")")
  def quiesce(self, sigID = 0, _d = 0):
    # Remove coincident events
    d = self.time2Index(_d, sigID)
    ok = wffunc.quiesce(self.onsets, self.epiID, d)
    for i in range(len(self.Data)):
      self.Data[i] = self.Data[i][ok]
    self.epiID = self.epiID[ok]
    self.onsets = self.onsets[ok]
    _active0 = self.active[0][ok]
    _active1 = self.active[1][ok]
    _active2 = self.active[2][ok]
    if self.trig is not None:
      cs = ok.cumsum() - 1
      self.trig[:,0] = cs[self.trig[:,0]]
      self.trig = self.trig[ok]
      if not(len(self.trig)):
        self.trig = None
        self.trigID = None
    self.setActive([_active0, _active1, _active2], len(_active0), True)
    return ok
  def align(self, sigID = 0, alignSpec = 0, _timeRange = None, _trigAl = None, _alignRange = None): 
    # alignRange: None/False = no limit, True = limit to timeRange,
    #       .. if float: limit to intra-episode interval fractile
    if type(_trigAl) is str: _trigAl = str2bool(_trigAl)
    alignRange = False if _alignRange is None else _alignRange
    trigAl = False if _trigAl is None else _trigAl
    timeRange = self.range2Indices(_timeRange)  
    timeRange = [timeRange[0], timeRange[1], None]
    dt = abs(timeRange[1] - timeRange[0])
    if type(alignRange) is bool:
      if alignRange is True:
        alignRange = [-dt, dt]
        timeRange[2] = tuple(alignRange)
    _trig, _onsets, _epiID = self.trig, self.onsets, self.epiID
    if alignSpec != 0:
      [_, I, ii] = wffunc.align(self.Data[sigID], alignSpec, timeRange)
    elif self.trig is None:
      return
    else:
      if _timeRange is None:
        ok = np.ones(len(self.trig), dtype = bool)
      else:
        ok = np.logical_and(self.trig[:,1]>=timeRange[0], self.trig[:,1]<timeRange[1])
      _trig = self.trig[ok]
      _onsets = self.onsets[_trig[:,0]]
      _epiID = self.epiID[_trig[:,0]]
      if isarray(alignRange):
        if elType(alignRange) is float:
          alignRange = np.array(np.array(alignRange, dtype = float) / self.SiGnOf[sigID][0])
        timeRange[2] = tuple(alignRange) 
      elif isnum(alignRange): # Not bool and not array
        if alignRange > 1.: alignRange *= 0.01
        if alignRange > 1.: raise ValueError("Unknown range specification: " + str(alignRAnge))
        ii0, ii1 = _trig[:,0], _trig[:,1]
        I = np.unique(ii0)
        di = []
        for i in I:
          k = np.nonzero(ii0 == i)[0]
          if len(k) > 1:
            di.append(np.diff(ii1[k]))
        if len(di):
          di = np.array(np.hstack(di), dtype = float)
          if alignRange < 1:
            dt = int(round(fractile(di, alignRange)))
          else:
            dt = int(max(di))
        timeRange[2] = (-dt, dt)
      [_, I, ii] = wffunc.align(self.Data[sigID], _trig, timeRange[2])
    if not(len(I)): return # let's not waste any time
    self.trig, self.onsets, self.epiID = _trig, _onsets, _epiID
    for i in range(len(self.Data)): # this method assumes identical samplint for all channels
      self.Data[i] = self.Data[i][I[0], I[1]]
    ii0, ii1 = ii[:,0], ii[:,1]
    self.onsets += np.ravel(I[1][:, 0])
    _active0 = self.active[0][ii0]
    _active1 = self.active[1][ii0]
    _active2 = self.active[2][ii0]
    self.setActive([_active0, _active1, _active2], len(_active0), True)
    if trigAl: # independent of alignment specification
      self.trig = np.empty((len(ii), 2), dtype = int)
      self.trig[:,0] = np.arange(len(ii))
      self.trig[:,1] = ii1[0] - I[1][0,0]
      self.trigID = sigID
    else:
      self.trigger(-1)
    self.quiesce()
    if _alignRange is None:
      if _timeRange is None:
        self.appendLog("align(" + str(sigID) + ", " + str(alignSpec) + ")")
      else:
        self.appendLog("align(" + str(sigID) + ", " + str(alignSpec) + ", " +
                       str(_timeRange) + ", " + str(trigAl) + ")")
    else:
      self.appendLog("align(" + str(sigID) + ", " + str(alignSpec) + ", " +
                     str(_timeRange) + ", " + str(trigAl) + ", " + str(_alignRange) + ")")

  def setTrig(self, _trigID = 0, _trig = None):
    self.trigID = max(0, _trigID)
    self.trig = _trig
  
    # Removed triggers from deleted:
    if self.trig is not None:
      if not(len(self.trig)):
        self.trig = None
    if self.trig is not None:
      i0 = self.trig[:,0]
      a0 = np.nonzero(self.active[0])[0]
      ok = np.zeros(len(i0), dtype = bool)
      for i in range(len(ok)):
        ok[i] = np.any(i0[i] == a0)
      self.trig = self.trig[ok,:]
      if not(len(self.trig)):
        self.trig = None
    if self.trig is None:
      self.trigID = None
  def trigger(self, sigID = 0, trigSpec = 0, trigVal = 0, _trigD = 0):
    self.trigID = sigID
    if self.trigID < 0: # call to delete all triggers
      return self.setTrig(-1)
    trigD = self.time2Index(_trigD, sigID)
    trigval = self.ADC2int(trigVal, sigID) 
    _trig = wffunc.trigger(self.Data[sigID], trigSpec, trigval, trigD)
    self.setTrig(sigID, _trig)
    if sigID >= 0:
      self.appendLog("trigger(" + str(sigID) + ", " + str(trigSpec) + ", " + str(trigVal) + ", " + str(trigD) + ")")  
  def baseline(self, sigID = 0, _timeRange = None, mode = None, fixc = None):
    if nDim(_timeRange) != 2:
      timeRange = self.range2Indices(_timeRange)
    else:
      timeRange = [[]] * len(_timeRange)
      for i in range(len(_timeRange)):
        timeRange[i] = self.range2Indices(_timeRange[i])
    self.Data[sigID] = wffunc.baseline(self.Data[sigID], timeRange, mode, fixc) 
    # We do not need to make any changes to self.active
    if mode is None:
      self.appendLog("baseline(" + str(sigID) + ", " + str(timeRange) + ")")    
    elif fixc is None:
      self.appendLog("baseline(" + str(sigID) + ", " + str(timeRange) + ", " + str(mode) + ")")    
    else:
      self.appendLog("baseline(" + str(sigID) + ", " + str(timeRange) + ", " + str(mode) + ", " + str(fixc) + ")")    
  def homogenise(self, sigID = 0, _timeRange = None, mode = None, overall = None):
    if nDim(_timeRange) != 2:
      timeRange = self.range2Indices(_timeRange)
    else:
      timeRange = [[]] * len(_timeRange)
      for i in range(len(_timeRange)):
        timeRange[i] = self.range2Indices(_timeRange[i])
    self.Data[sigID] = wffunc.homogenise(self.Data[sigID], timeRange, mode, overall) 
    # We do not need to make any changes to self.active
    if mode is None:
      self.appendLog("homogenise(" + str(sigID) + ", " + str(timeRange) + ")")    
    elif overall is None:
      self.appendLog("homogenise(" + str(sigID) + ", " + str(timeRange) + ", " + str(mode) + ")")    
    else:
      self.appendLog("homogenise(" + str(sigID) + ", " + str(timeRange) + ", " + str(mode) + ", " + str(overall) + ")")    
  def extract(self, sigID = 0, _timeRange = None, notlog = False): # outputs a trimmed data set without changing it
    timeRange = self.range2Indices(_timeRange)
    self.extracted = np.array(self.Data[sigID][:, timeRange[0]:timeRange[1]], dtype = float)
    if self.SiGnOf[sigID][1] != 1.: self.extracted *= self.SiGnOf[sigID][1]
    if self.SiGnOf[sigID][2] != 0.: self.extracted += self.SiGnOf[sigID][0]
    if not(notlog): self.appendLog("extract(" + str(sigID) + ", " + str(timeRange) + ")")    
    self.Results = np.copy(self.extracted)
    return self.Results
  def export(self, _fn, onsOpts = 2, wavOpts = 2, ch = None, win = None, active0 = None): # this is never logged
    if active0 is None: active = self.active[0]
    tdf = TDF()
    tdf.setData(self.Data, self.DataInfo, self.onsets)
    tdf.writeData(_fn, onsOpts, wavOpts, ch, win, argtrue(active0))
    tdf.clearData()
  def updatetable(self, _Results = [], _Labels = [], _Active = None, _EpiID = None, _Tonsets = None, opts = 0): # opts: None = no change, 0 = unkilled, 1 = selected    
    self.Results, self.Labels = _Results, _Labels
    if not(len(self.Results)): return
    if not(len(self.Labels)): return
    Active = self.active if _Active is None else _Active
    EpiID = self.epiID  if _EpiID is None else _EpiID
    Tonsets = np.array(self.onsets, dtype = float) * float(self.SiGnOf[0][0]) if _Tonsets is None else _Tonsets
    if type(self.Results) is np.ndarray:
      if nDim(self.Results) == 1:
        self.Results = self.Results.reshape( (len(self.Results), 1) )
    n = np.sum(Active[0])
    Res = [[]] * n
    Lbl = ["ID"] + ["Active"] + ["Episode"] + ["Onset"]
    nLbl = len(Lbl)
    Lbl += self.Labels
    k = -1
    for i in range(len(self.Results)):
      if Active[0][i]:
        k += 1
        Res[k] = [[]] * (len(self.Results[i]) + nLbl)
        Res[k][0] = i
        Res[k][1] = Active[1][i]
        Res[k][2] = EpiID[i]
        Res[k][3] = Tonsets[i]
        for j in range(len(self.Results[i])):
          Res[k][j+nLbl] = self.Results[i][j]
    self.table.setData(Res, Lbl, 0) 
  def writetable(self, _opfn = None, _opdn = None, _comm = None):
    if type(_comm) is str:
      if not(len(_comm)):
        _comm = None
    opfn, opdn = _opfn, _opdn
    ipfn, ipdn = self.stpf, self.dirn + "/"
    if opdn is None:
      opdn = ipdn.replace('data', 'analyses')
      if not(os.path.exists(opdn)):
        opdn = ipdn
    if opfn is None or not opdn: 
      opfn = ipfn + DEFAULT_TDF_EXTENSION
    if opdn[-1] != '/':
      opdn += '/'
    if _comm is None:
      if _opdn is None:
        if _opfn is None:
          self.appendLog("writetable()")
        else:   
          self.appendLog("writetable('" + _opfn + "')")
      else:
        self.appendLog("writetable('" + _opfn + "', '" + _opdn + "')")  
    else:
      self.appendLog("writetable('" + _opfn + "', '" + _opdn + "', '" + _comm + "')")  
    if os.path.exists(opdn+opfn):
      Table = iofunc.readDTFile(opdn+opfn)
      Table += [tuple(self.log)]
      Table += [self.table.X]
    else:
      Table = [tuple(self.log), self.table.X]
    if self.writeDisabled is True: return
    iofunc.writeDTFile(opdn+opfn, Table) 
  def setActive(self, _active = None, _nepis = None, notlog = False):
    if _active is None: _active = [None, None, None]
    if len(_active) == 1: _active = [_active[0], None]
    if len(_active) == 2: _active = [_active[0], _active[1], None]
    if _nepis is None: _nepis = self.nepis
    if self.active is not None:
      if self.active.shape[1] != _nepis:
        self.active = None
    if self.active is None or _nepis != self.nepis:
      self.nepis = _nepis
      self.active = np.ones((3, self.nepis), dtype = bool)
      self.active[2,:] = np.zeros(self.nepis, dtype = bool)    
    elif type(_active) is np.ndarray:
      if nDim(_active) == 2:
        if len(_active) == 3:
          self.active = np.copy(_active)
    actstr = ["None", "None", "None"]
    for i in range(2): # we don't really care which traces are highlighted
      activei = _active[i]
      if type(activei) is bool:           
        self.active[i] = np.tile(activei, self.nepis)
        actstr[i] = str(sifunc.bool2uint(self.active[i,:], list))
      elif elType(activei) is bool:
        self.active[i] = np.array(activei, dtype = bool)
        actstr[i] = str(sifunc.bool2uint(self.active[i,:], list))
      elif elType(activei) is int:
        actstr[i] = str(activei)
        self.active[i] = sifunc.uint2bool(activei, self.nepis)
    self.active[1] = np.logical_and(self.active[0], self.active[1])
    '''
    if trig is not None:
      ok = np.zeros(len(trig))
      for i in range(len(trig)):
        ok[i] = self.active[0][trig[i,0]]
      self.trig = self.trig[ok, :]
      if not(len(self.trig)): self.trig = None
    '''
    if notlog: return
    activeStr = "setActive([" + actstr[0] + ", " + actstr[1] + "], " + str(_nepis) + ")"
    if np.all(self.active[0] == self.active[1]):
      activeStr = "setActive([" + actstr[0] + ", True], " + str(_nepis) + ")"
    if activeStr != self.log[-1]: 
      self.appendLog(activeStr)
  def offset(self, sigID = 0, ind = None, _timeRange = None):
    if ind is None: ind = self.active[2]
    if elType(ind) is bool: ind = argtrue(ind)
    if nDim(_timeRange) != 2:
      timeRange = self.range2Indices(_timeRange)
    else:
      timeRange = [[]] * len(_timeRange)
      for i in range(len(_timeRange)):
        timeRange[i] = self.range2Indices(_timeRange[i])
    indstr = str(sifunc.bool2uint(self.active[2,:], list))
    self.Data[sigID] = wffunc.offsetmean(self.Data[sigID], ind, timeRange) 
    self.appendLog("offset(" + str(sigID) + ", [" + indstr + "], " + str(timeRange) + ")")    
  def psa(self, sigID = 0, _timeRange = None, irange = None, _u = None, _pgb = None):  
    _Data = self.extract(sigID, _timeRange, True)
    self.anal = wfanal.synResponse()
    self.anal.analyse(_Data, self.SiGnOf[sigID][0], irange, _u, _pgb)
    _Results = np.copy(self.anal.Z)
    _Labels = self.anal.lbl[:]
    self.updatetable(_Results, _Labels)
    if _u is None:
      if irange is None:
        if _timeRange is None:
          self.appendLog("psa(" + str(sigID) + ")")
        else:   
          self.appendLog("psa(" + str(sigID) + ", " + str(_timeRange) + ")")
      else:
        self.appendLog("psa(" + str(sigID) + ", " + str(_timeRange) + ", " + str(irange) + ")")  
    else:
        self.appendLog("psa(" + str(sigID) + ", " + str(_timeRange) + ", " +
                       str(irange) + ", " + str(_u) + ")")  
    return self.Results, self.Labels
  def expfit(self, sigID = 0, _timeRange = None, _pgb = None):  
    _Data = self.extract(sigID, _timeRange, True)
    self.anal = wfanal.expDecay()
    self.anal.analyse(_Data, self.SiGnOf[sigID][0], _pgb)
    _Results = np.copy(self.anal.Z)
    _Labels = self.anal.lbl[:]
    self.updatetable(_Results, _Labels)
    if _timeRange is None:
      self.appendLog("expfit(" + str(sigID) + ")")
    else:   
      self.appendLog("expfit(" + str(sigID) + ", " + str(_timeRange) + ")")
    return self.Results, self.Labels
  def mwd(self, sigID = 0, _timeRange = None):
    _Data = self.extract(sigID, _timeRange, True)
    _Results = np.mean(_Data, axis = 1)
    _Labels = ['MWD']
    self.updatetable(_Results.reshape((len(_Results), 1)), _Labels)
    if _timeRange is None:
      self.appendLog("mwd(" + str(sigID) + ")")
    else:   
      self.appendLog("mwd(" + str(sigID) + ", " + str(_timeRange) + ")")
    return self.Results, self.Labels
  def triggers(self, sigIDs = None, _timeRange = None, notlog = False): # non-overlapping trigger analysis
    if sigIDs is None: sigIDs = 0;
    if isint(sigIDs): sigIDs = [int(bool(sigIDs)), int(not(bool(sigIDs)))]
    sigID = sigIDs[0]
    timeRange = self.range2Indices(_timeRange, sigID)
    if self.trig is None: raise ValueError("Cannot perform trigger analysis without triggers")
    ok = np.nonzero(np.logical_and(self.trig[:, 1] >= timeRange[0], self.trig[:, 1] < timeRange[1]))[0]
    if not(len(ok)): return None
    _trig = self.trig[ok, :]
    trig0 = _trig[:, 0]
    trig1 = _trig[:, 1]
    _Lbl = ['Times']
    _Res = np.array(self.onsets[trig0] + trig1, dtype = float) * float(self.SiGnOf[sigID][0])
    window = [timeRange[0] - trig1.min(), timeRange[1] - trig1.max()]
    K = np.unique(trig0)
    for k in K:
      i = np.nonzero(trig0 == k)[0]
      if len(i) > 1:
        hmi = np.diff(trig1[i]).min()/2
        window = [max(window[0], -hmi), min(window[1], hmi)]
    _uiData, I, ii = wffunc.align(self.Data[sigID], self.trig[ok], tuple(window))
    uiData = np.array(_uiData, dtype = float)
    if self.SiGnOf[self.trigID][1] != 1.: uiData *= self.SiGnOf[self.trigID][1]
    if self.SiGnOf[self.trigID][2] != 0.: uiData += self.SiGnOf[self.trigID][2]
    if len(I):         # only align if alignment was actually performed
      ii0, ii1 = ii[:, 0], ii[:, 1]
      _active0 = self.active[0][ii0]
      _active1 = self.active[1][ii0]
      _active2 = self.active[2][ii0]
      _Active = np.array([_active0, _active1, _active2])
      _EpiID = self.epiID[trig0]
      _Res0 = _Res.reshape((len(_Res), 1))
      _Onsets = self.onsets[trig0] + np.ravel(I[1][:,0])
      #_Tonsets = np.array(_Onsets, dtype = float) * float(self.SiGnOf[sigID][0])
      #self.updatetable(_Res, _Lbl, _Active, _EpiID, _Tonsets)
      _Lbl.append("BaseCom")
      _Lbl.append("StepCom")
      _Lbl.append("StepRec")
      _Res1 = np.array(self._Data[sigIDs[1]][_EpiID, 0], dtype = float)
      _Res2 = np.array(self.Data[sigIDs[1]][_trig[:,0], _trig[:,1]], dtype = float)
      _Res3 = np.array(self.Data[sigIDs[0]][_trig[:,0], _trig[:,1]], dtype = float)
      if self.SiGnOf[sigIDs[1]][1] != 1.: _Res1 *= self.SiGnOf[sigIDs[1]][1]; _Res2 *= self.SiGnOf[sigIDs[1]][1]
      if self.SiGnOf[sigIDs[1]][2] != 0.: _Res1 += self.SiGnOf[sigIDs[1]][2]; _Res2 += self.SiGnOf[sigIDs[1]][2]
      if self.SiGnOf[sigIDs[0]][1] != 1.: _Res3 *= self.SiGnOf[sigIDs[0]][1];
      if self.SiGnOf[sigIDs[0]][2] != 0.: _Res3 += self.SiGnOf[sigIDs[0]][2];
      _Res = np.vstack( (np.ravel(_Res0), _Res1, _Res2, _Res3) ).T
      _Tonsets = np.array(_Onsets, dtype = float) * float(self.SiGnOf[sigIDs[0]][0])
      self.updatetable(_Res, _Lbl, _Active, _EpiID, _Tonsets)
      if not(notlog):
        if _timeRange is None:
          self.appendLog("triggers(" + str(sigID) + ")")
        else:   
          self.appendLog("triggers(" + str(sigID) + ", " + str(_timeRange) + ")")
      return _Res, _Lbl, _Active, _EpiID, _Onsets, uiData, I, ii, _trig
    return None
  def spikeShape(self, sigID = 0, _timeRange = None, _pgb = None):  
    _Data = self.extract(sigID, _timeRange, True)
    self.anal = wfanal.spikeShape()
    self.anal.analyse(_Data, self.SiGnOf[sigID][0], _pgb)
    self.updatetable(self.anal.Z, self.anal.lbl)
    if _timeRange is None:
      self.appendLog("spikeShape(" + str(sigID) + ")")
    else:   
      self.appendLog("spikeShape(" + str(sigID) + ", " + str(_timeRange) + ")")
    return self.Results, self.Labels
  def iStep(self, sigIDs = 0, _timeRange = None, _pgb = None):  
    if sigIDs is None: sigIDs = 0;
    if isint(sigIDs): sigIDs = [int(bool(sigIDs)), int(not(bool(sigIDs)))]
    _Data1 = self.extract(sigIDs[1], _timeRange, True)
    _Data0 = self.extract(sigIDs[0], _timeRange, True)
    self.anal = wfanal.stepResponse()
    self.anal.analyse(_Data0, _Data1, self.SiGnOf[sigIDs[0]][0], _pgb)
    self.updatetable(self.anal.Z, self.anal.lbl)
    if _timeRange is None:
      self.appendLog("iStep(" + str(sigIDs) + ")")
    else:   
      self.appendLog("iStep(" + str(sigIDs) + ", " + str(_timeRange) + ")")
    return self.Results, self.Labels
  def vStep(self, sigIDs = 0, _timeRange = None, _pgb = None):  
    if sigIDs is None: sigIDs = 0;
    if isint(sigIDs): sigIDs = [int(bool(sigIDs)), int(not(bool(sigIDs)))]
    _Data1 = self.extract(sigIDs[1], _timeRange, True)
    _Data0 = self.extract(sigIDs[0], _timeRange, True)
    self.anal = wfanal.stepBiexp()
    self.anal.analyse(_Data0, _Data1, self.SiGnOf[sigIDs[0]][0], _pgb)
    self.updatetable(self.anal.Z, self.anal.lbl)
    if _timeRange is None:
      self.appendLog("vStep(" + str(sigIDs) + ")")
    else:   
      self.appendLog("vStep(" + str(sigIDs) + ", " + str(_timeRange) + ")")
    return self.Results, self.Labels
  def setLabel(self, _label = "", notlog = False):
    if len(_label): self.lable = _label
    if notlog or not(len(_label)): return
    self.appendLog("setLabel('" + str(_label) + "')")
  def setComment(self, _comment = None):
    self.comment = _comment
    
class Pyclamp (pyclamp): # front end
  btnWidth = 100
  deftifwin = 0.95 # default trigger interval fractile window
  Form = None
  Area = None
  Bbox = None
  Dock = None
  SetAnal = None
  menuBar = None
  chan = None
  chanNames = None
  times = None
  uiID = None
  uiIP = None
  maxEpiSel = 0  
  cbdata = None
  def __init__ (self, _Form = None):
    self.initialise()
    self.setForm(_Form)
  def setForm(self, _Form = None):
    if self.Form is None:
      if _Form is None: return 
      self.Form = _Form
    if self.Area is None: self.Area = area()
    self.setArea(self.Area)
  def setArea(self, _Area = None):
    if _Area is not None: self.Area = _Area
    if self.Form is None or self.Area is None: return
    try:
      self.Form.Widget = self.Form.FWidget
    except AttributeError:
      self.Form.FWidget = self.Form
    try:
      self.Form.FWidget.setCentralWidget(self.Area)
    except AttributeError:
      self.Form.FWidget.setWidget(self.Area)
  def close(self):
    if self.Form is None: return
    self.Form.close()
  def readData(self):
    self.Dlg = mio.mfilioGUI()
    self.Dlg.Open(self.readDataOK, self.readDataCC)
  def readDataOK(self, event = None):
    dlgIP = self.Dlg.CloseDlg(1)    
    # Close the dialog and only then instantiate a form
    if dlgIP is None: return False
    if self.Form is None: self.Form = lbw.LBWidget(None, None, None, 'form')
    self.setForm(self.Form)
    dirn = dlgIP[0]
    filn = dlgIP[1]
    chno = dlgIP[2]
    sigo = dlgIP[3]
    self.chan = dlgIP[4]
    self.setDataDir(dirn)
    self.setDataFiles(filn)
    self.setChannels(self.chan)
    self.readIntData()      
    self.SetSelection() # legacy facility to fine-tune selection
  def readDataCC(self, event = None):
    dlgIP = self.Dlg.CloseDlg(0)    
  def InitGUI(self):
    if self.chan is not None:
      self.chanNames = []
      for ch in self.chan:
        self.chanNames.append(ch.name)          
    else:
      self.chanNames = [''] * self.nchan
      for i in range(self.nchan):
        self.chanNames[i] = "Channel " + str(i)
    return self.SetData()
  def SetSelection(self, event = None):
    selection = []
    for intdata in self.IntData[0]: selection.append(-len(intdata))
    self.setSelection(selection)
    if self.P is not None:
      self.P.terminate()
      self.P.join()
      del self.P
      self.P = None
    gc.collect()
    self.SetClampConfig()
  def SetClampConfig(self):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Set passive channel")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.BG0 = lbw.BGWidgets(self.Dlg, 'Clamp type', 0, ["Current clamp", "Voltage clamp"], 'radiobutton', 0) 
    self.BG1 = lbw.BGWidgets(self.Dlg, 'Configuration', 0, ["Whole cell", "Patched membrane"], 'radiobutton', 0)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.SetClampConfigOK)
    self.BBx.Widgets[1].connect("btndown", self.SetClampConfigCC)            
    self.Box.add(self.BG0)
    self.Box.add(self.BG1)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()  
  def SetClampConfigOK(self, event = None):  
    uiClamp = self.BG0.retData()
    uiConfig = self.BG1.retData()
    self.DlgClose()
    self.setClampConfig(uiClamp, uiConfig)
    return self.InitGUI()
  def SetClampConfigCC(self, event = None):
    self.DlgClose()
    return False
  def SetActive(self, _uiID = None, _notlog=None):
    if _uiID is None: _uiID = 0
    if type(_uiID) is list or type(_uiID) is np.ndarray:
      for _pw in self.SetAnal.pw:
        _pw.setactive(_uiID)
      _uiID = 0
    _active = self.SetAnal.pw[_uiID].active
    nactive = len(_active[0])
    notlog = np.array(_active[0]).sum() == nactive and np.array(_active[1]).sum() == nactive if _notlog is None else _notlog
    self.setActive(_active, nactive, notlog)
    self.SetTrig() # implement deletions on self.trig
  def SetData(self, _overlay = None, _lims = None, _active = None): # this function doesn't change self.Data
    if type(self.Data) is list:
      self.nchan, self.nepis, self.nsamp = len(self.Data), len(self.Data[0]), len(self.Data[0][0])
    else:
      self.nchan, self.nepis, self.nsamp = self.Data.shape
    _Active = _active
    if _Active is None: _Active = self.active
    if self.SetAnal is None:
      self.SetAnal = pywav(self.Data, self.DataInfo, self.onsets, _Active)
      self.SetAnal.setPlots(parent = self.Area)
      self.SetMenus()
      self.SetButtons()
      self.Form.show()
      self.Form.Widget.raise_()
      self.Form.Widget.activateWindow()
      self.Form.Widget.showNormal()
    elif _active is not None: # specifying _active forces data update
      self.SetAnal.updateData(self.Data, self.DataInfo, self.onsets, _Active)
    self.SetTrig(self.trigID, self.trig)
    num = np.array(self.SetAnal.pw[0].active[1]).sum() 
    den = np.array(self.SetAnal.pw[0].active[0]).sum() 
    trg = 0 if self.trig is None else len(self.trig)
    self.Form.Widget.setWindowTitle(self.stpf + " [" + str(num) + '/' + str(den) + " (" + str(trg) + ")]")
    if _overlay is not None: self.SetAnal.pw[0].setOverlay(_overlay) # one does the lot
    if _lims is not None: self.SetAnal.resetLims(_lims)
    self.SetAnal.setFocus()
  def SetTrig(self, _trigID = None, _trig = None):
    if _trigID is None and _trig is None and self.trigID is not None and self.SetAnal is not None: # flags update self.trig
      _trig = self.SetAnal.pw[self.trigID].marks
      if _trig is not None: 
        self.setTrig(self.trigID, _trig.T)
      else:
        self.trigger(-1)
      _trig = self.trig     # impose any deletions onto plot
      _trigID = self.trigID
    self.trigID = _trigID
    self.trig = _trig
    if self.trig is not None:
      marks = [None] * self.nchan
      markc = 'T'
      markr = 1.025
      marks[self.trigID] = np.array(self.trig, dtype = int).T
      self.SetAnal.setMarks(marks, markc, markr)
      self.SetAnal.pw[self.trigID].setWave()
  def SetMenus(self, enabled = None):
    newMenu = False
    if self.menuBar is None:
      if enabled is not None: return
      self.menuBar = self.Form.FWidget.menuBar()
      newMenu = True
    if enabled is not None: self.menuBar.setEnabled(enabled)
    if not(newMenu): return
    self.menuTitl = ['&Selection', '&Process', '&Analysis', '&Output']
    M = len(self.menuTitl)
    self.menuList = [[]] * M
    self.menuText = [[]] * M
    self.menuCall = [[]] * M
    self.menuSTip = [[]] * M
    self.menuSCKB = [[]] * M
    self.menuActn = [[]] * M
    self.menuText[0] = ['&Select', 'Discriminate_&2D', 'Discriminate_&6D']
    self.menuCall[0] = [self.Select, self.Disc1, self.Disc3]
    self.menuSTip[0] = ['Select episodes', 'Discriminate using 2 parameters', 'Discriminate using 6 parameters']
    self.menuSCKB[0] = ['Ctrl+e', 'Ctrl+2', 'Ctrl+6']
    self.menuText[1] = ['&Restore', 'Tri&m', '&Filter', 'Trigger', 'A&lign', '&Baseline', '&Offset', '&Homogenise']
    self.menuCall[1] = [self.Restore, self.Trim, self.Filter, self.Trigger, self.Align, self.Baseline, self.Offset, self.Homogenise]
    self.menuSTip[1] = ['Restore data', 'Reduce wave data', 'Filter wave data', 'Trigger on wave data', 'Align excerpts at point', 'Baseline wave data to common window', 'Offset wave data against average of selected', 'Shift and scale wave data to fixed window range']
    self.menuSCKB[1] = ['Ctrl+z', 'Ctrl+m', 'Ctrl+f', 'Ctrl+t', 'Ctrl+l', 'Ctrl+b', 'Ctrl+o', 'Ctrl+h']
    self.menuText[2] = ['&Triggers', 'Spi&ke shape', '&Command step', 'E&xponential fit', '&Post-synaptic response', '&Mean windowed discursion']
    self.menuCall[2] = [self.Triggers, self.SpikeShape, self.Step, self.ExpFit, self.PSA, self.MWD]
    self.menuSTip[2] = ['Tabulate trigger times', 'Analyse spike shape', 'Analyse IV steps', 'Fit exponential decays', 'Analyse post-synaptic response', 'Measure windowed mean peak/trough']
    self.menuSCKB[2] = ['Ctrl+g', 'Ctrl+k', 'Ctrl+c', 'Ctrl+x', 'Ctrl+p', 'Ctrl+w']
    self.menuText[3] = ['Export a&s text file', 'Outp&ut log to console']
    self.menuCall[3] = [self.Export, self.printLog]
    self.menuSTip[3] = ['Export traces as tab delimited text file', 'Display log on console if available']
    self.menuSCKB[3] = ['Ctrl+s', 'Ctrl+u']
    for i in range(M):
      self.menuList[i] = self.menuBar.addMenu(self.menuTitl[i])
      m = len(self.menuText[i])
      self.menuActn[i] = [[]] * m
      for j in range(len(self.menuText[i])):
        self.menuActn[i][j] = QtGui.QAction(QtGui.QIcon('open.png'), self.menuText[i][j], self.Form.FWidget)        
        self.menuActn[i][j].setShortcut(self.menuSCKB[i][j])
        self.menuActn[i][j].setStatusTip(self.menuSTip[i][j])
        self.menuActn[i][j].triggered.connect(self.menuCall[i][j])
        self.menuList[i].addAction(self.menuActn[i][j])
  def SetButtons(self): 
    if self.Dock is not None or self.Bbox is not None: self.ClrButtons()
    I = 1
    self.Dock = [None] * I
    self.Bbox = [None] * I
    for i in range(I):
      self.Dock[i] = dock()
      self.Bbox[i] = self.Dock[i].addBbox()
      self.Area.add(self.Dock[i], 'bottom')
      if i == 0:
        btnText = ['Restore', 'Trim', 'Filter', 'Trigger', 'Align', 'Baseline']
        btnCall = [self.Restore, self.Trim, self.Filter, self.Trigger, self.Align, self.Baseline]
        for j in range(len(btnText)):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnWidth)
          self.Bbox[i].setText(j, btnText[j])
          self.Bbox[i].Connect(j, btnCall[j])
      '''  
      elif i == 1:
        btnText = ['Triggers', 'Spike shape', 'Step', self.pstext, 'MWD', 'Export']
        btnCall = [self.Triggers, self.SpikeShape, self.Step, self.PSA, self.MWD, self.Export]
        for j in range(len(btnText)):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnWidth)
          self.Bbox[i].setText(j, btnText[j])
          self.Bbox[i].Connect(j, btnCall[j])
      elif i == 2:
        btnText = ['Log', 'Disc_2D', 'Disc_6D', 'Select', 'Offset', 'Homogenise']
        btnCall = [self.printLog, self.Disc1, self.Disc3, self.Select, self.Offset, self.Homogenise]
        for j in range(len(btnText)):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnWidth)
          self.Bbox[i].setText(j, btnText[j])
          self.Bbox[i].Connect(j, btnCall[j])
      '''
  def SetDocks(self, show = None):
    if show is None: return
    if show:
      if self.Dock is not None:
        for _Dock in self.Dock:
          _Dock.show()
      if self.SetAnal is not None:
        for _dock in self.SetAnal.docks:
          _dock.show()
    else:
      if self.Dock is not None:
        for _Dock in self.Dock:
         _Dock.hide()
         _Dock.resize(0,0)
      if self.SetAnal is not None:
        for _dock in self.SetAnal.docks:
          _dock.hide()
          _dock.resize(0,0)
  def SetGUI(self, show = None):
    self.SetMenus(show)
    self.SetDocks(show)
  def DlgClose(self, event = None):
    self.Dlg.close()
    del self.Dlg
    self.Dlg = None
    if self.SetAnal is None: return
    self.SetAnal.setFocus()
  def Restore(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Restore")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    lorestore = len(self.log) > self.ccl+2
    if lorestore: # only offer if _more_ than one action has been logged
      self.uilog = self.log[self.ccl+1:][:]
      self.LBi = lbw.LBWidget(self.Dlg, "If restoring from log, select final line", 1,"listbox", len(self.uilog)-2, self.uilog)
      #self.LBi.setMode(3, range(self.NX))
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button", "Button"], ["Restore from log", "Restore completely", "Cancel"]) 
      self.BBx.Widgets[0].connect("btndown", self.RestoreLO)
      self.BBx.Widgets[1].connect("btndown", self.RestoreOK)
      self.BBx.Widgets[2].connect("btndown", self.DlgClose)
      self.Box.add(self.LBi)
    else:
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Restore completely", "Cancel"]) 
      self.BBx.Widgets[0].connect("btndown", self.RestoreOK)
      self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.overlay = self.SetAnal.pw[0].overlay
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def RestoreOK(self, ev = None):
    self.cclog = self.log[self.ccl]
    self.DlgClose()
    self.setSelection() # restore wave data    
    self.trigger(-1)    # remove all triggers - log unaffected
    self.runstr(self.cclog) # set clamp config to as before
    self.SetData(self.overlay, None, self.active) # no axis limit preservation (not even y)   
  def RestoreLO(self, ev = None):
    self.cclog = self.log[self.ccl]
    lbi = self.LBi.retData()[0]+1
    self.uilog = self.uilog[:lbi]
    self.DlgClose()
    self.setSelection() # restore wave data    
    self.trigger(-1)    # remove all triggers - log unaffected
    _writeDisabled = self.writeDisabled
    self.writeDisabled = True
    self.runstr(self.cclog)
    self.runstr(self.uilog)
    self.writeDisabled = _writeDisabled
    self.SetData(self.overlay, None, self.active) # no axis limit preservation (not even y)   
  def Trim(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Trim")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.TrimOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def TrimOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.DlgClose()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.SetAnal.setCursor(self.uiID, [0, 0], self.TrimCB)
  def TrimCB(self, ev = None):  
    if ev.cursors is None: return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.SetActive(self.uiID) # log any deletions
    self.trim(self.uiID, win, self.overlay)
    self.SetData(self.overlay, None, self.active) # no axis limit preservation (not even y)   
  def Filter(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Filter")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Bandpass', 0, ["Low-pass", "High-pass"], 'radiobutton', 0)
    self.EDx = [[]] * 3
    self.EDx[0] = lbw.LBWidget(self.Dlg, 'Corner frequency (Hz): ', 1, 'edit', str(int(round(0.2/self.SiGnOf[0][0]))))
    self.EDx[1] = lbw.LBWidget(self.Dlg, 'Transition bandwidth (%): ', 1, 'edit', str(50))
    self.EDx[2] = lbw.LBWidget(self.Dlg, 'Transition non-linearity (%): ', 1, 'edit', str(2))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.FilterOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BGB)
    self.Box.add(self.EDx[0])
    self.Box.add(self.EDx[1])
    self.Box.add(self.EDx[2])
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def FilterOK(self, ev = None):
    for i in range(3):
      OK = self.EDx[i].valiData(1, 1)
      if not(OK[0]) or not(OK[1]): return
    self.uiID = self.CBx.retData()
    self.uiIP = self.BGB.retData()
    freq = np.fabs(float(self.EDx[0].retData()))
    perc = np.fabs(float(self.EDx[1].retData()))
    nlin = np.fabs(float(self.EDx[2].retData()))
    mode = self.uiIP != 0
    freq = [freq, freq]
    if mode: # HPF
      freq[1] *= (1. - perc / 100.)
    else:    # LPF
      freq[1] *= (1. + perc / 100.)
    self.DlgClose()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.Lims = self.SetAnal.storeLims([0, 2], self.uiID)
    self.SetActive(self.uiID) # log any deletions
    _pgb = pgb()
    self.filter(self.uiID, mode, freq, nlin, _pgb)
    self.SetData(self.overlay, self.Lims, self.active) 
  def Trigger(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Trigger")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Trigger type', 0, 
               ["Ascent", "Descent", "2-point inflection", "3-point inflection"], 'radiobutton', 0)
    self.EDx = lbw.LBWidget(self.Dlg, 'Quiescent duration (s): ', 1, 'edit', '0')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.TriggerOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.EDx)
    self.Box.add(self.BGB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def TriggerOK(self, ev = None):
    OK = self.EDx.valiData(1, 1)
    if not(OK[0]) or not(OK[1]): return
    self.uiID = self.CBx.retData()
    self.uiIP = self.BGB.retData()
    self.uiQD = self.EDx.retData()
    self.DlgClose()    
    if self.uiIP < 2:
      self.SetAnal.pw[self.uiID].plot.setCursor([1], self.TriggerCB)
    else:
      self.SetAnal.pw[self.uiID].plot.setCursor([2] * self.uiIP, self.TriggerCB)
  def TriggerCB(self, ev = None):  
    if ev.cursors is None: return
    if self.uiIP < 2:
      trigSpec = -1 if self.uiIP else 1    
      trigLevel = ev.cursors[0]
    else:
      x = np.array([curs[0] for curs in ev.cursors])
      y = np.array([curs[1] for curs in ev.cursors])
      i = np.argsort(x)
      x, y = x[i], y[i]
      x = np.array( (x - x[0])/self.SiGnOf[self.uiID][0], dtype = int)
      trigSpec = list(x[1:] - x[0])
      trigLevel = list(np.diff(y))
    self.SetAnal.setCursor(self.uiID) # remove cursor(s)
    self.SetActive(self.uiID) # log any deletions
    self.trigger(self.uiID, trigSpec, trigLevel, float(self.uiQD))
    self.SetData()   
  def Align(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Align")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.AlTrig = None
    icstr = 'Interval centile (' + str(int(self.deftifwin*100.)) + '%)'
    Wopts = ['Episode width', 'Selected window', icstr, 'Longest interval']
    if self.trig is None:
      self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
      self.BGB = lbw.BGWidgets(self.Dlg, 'Alignment', 0, ["Maximum", "Minimum"], 'radiobutton', 0)
      self.RBx = lbw.LBWidget(self.Dlg, 'Set triggers by alignment:', 0, 'radiobutton', 0)
      self.WBx = lbw.LBWidget(self.Dlg, 'Limit peri-alignment window:', 1, 'combobox', 0, Wopts)
    else:
      self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', self.trigID, self.chanNames)
      self.BGB = lbw.BGWidgets(self.Dlg, 'Alignment', 0, ["Maximum", "Minimum", "Trigger"], 'radiobutton', 2)
      self.RBx = lbw.LBWidget(self.Dlg, 'Set triggers by alignment:', 0, 'radiobutton', True)
      self.WBx = lbw.LBWidget(self.Dlg, 'Limit peri-alignment window:', 1, 'combobox', 3, Wopts)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.AlignOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BGB)
    self.Box.add(self.RBx)
    self.Box.add(self.WBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def AlignOK(self, ev = None):
    uiwlmap = {0:None, 1:True, 2:self.deftifwin, 3:1}
    self.uiID = self.CBx.retData()
    self.uiAL = self.RBx.retData()
    self.uiWL = uiwlmap[self.WBx.retData()]
    self.uiIP = self.BGB.retData()
    self.DlgClose()    
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.AlignCB)
  def AlignCB(self, ev = None):
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    specmap = {0:+1, 1:-1, 2:0}
    alignSpec = specmap[self.uiIP]
    '''
    if alignSpec == 0:
      if self.uiID != self.trigID:
        self.uiID = self.trigID
    '''
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.Lims = self.SetAnal.storeLims([1, 1], self.uiID)
    self.SetActive(self.uiID) # log any deletions
    self.align(self.uiID, alignSpec, win, self.uiAL, self.uiWL)
    self.SetData(self.overlay, self.Lims, self.active) 
  def Baseline(self, ev = None):
    opttxt = ["Arithmetic mean", 
              "Single exponential decay fit (variable offset)", "Single exponential decay fit (zero offset)",
              "Double exponential decay fit (variable offset)", "Double exponential decay fit (zero offset)",
              "First derivative (single separation)",           "First derivative (double separation)"]
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Baseline")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.CBt = lbw.LBWidget(self.Dlg, 'Subtraction: ', 1, 'combobox', 0, opttxt)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Window', 0, ["Single", "Double"], 'radiobutton', 0)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.BaselineOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.CBt)
    self.Box.add(self.BGB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def BaselineOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.uiBT = self.CBt.retData()
    self.uiBG = self.BGB.retData()
    self.DlgClose()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    xcurs = [0, 0, 0, 0] if self.uiBG else [0, 0]
    self.SetAnal.setCursor(self.uiID, xcurs, self.BaselineCB)
  def BaselineCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    ev.cursors = np.sort(ev.cursors)
    win = [[ev.cursors[0], ev.cursors[1]], [ev.cursors[2], ev.cursors[3]]] if self.uiBG else [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    blt, blc = [None, 1, 1, 2, 2, -1, -2], [None, None, 0, None, 0, None, None]
    self.Lims = self.SetAnal.storeLims([0, 2], self.uiID)
    self.SetActive(self.uiID)       # store any deletions
    self.baseline(self.uiID, win, blt[self.uiBT], blc[self.uiBT])
    self.SetData(self.overlay, self.Lims, self.active)    
  def Homogenise(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Baseline")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Window', 0, ["Single", "Double"], 'radiobutton', 0)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.BaselineOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Homogenise")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BG0 = lbw.BGWidgets(self.Dlg, 'Match: ', 0, ["Minimum", "Range", "Maximum"], 'radiobutton', 1)
    self.BG1 = lbw.BGWidgets(self.Dlg, 'Homogeniety template: ', 0, ["By sweep", "Overall"], 'radiobutton', 0)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.HomogeniseOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BG0)
    self.Box.add(self.BG1)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()
  def HomogeniseOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.uiB0 = self.BG0.retData()
    self.uiB1 = self.BG1.retData()
    self.DlgClose()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.HomogeniseCB)
  def HomogeniseCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    ev.cursors = np.sort(ev.cursors)
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    b0, b1 = [-1, 0, +1], [False, True]
    self.Lims = self.SetAnal.storeLims([0, 2], self.uiID)
    self.SetActive(self.uiID)       # store any deletions
    self.homogenise(self.uiID, win, b0[self.uiB0], b1[self.uiB1])
    self.SetData(self.overlay, self.Lims, self.active)    
  def Disc1(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Discriminate")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.cbdata = [pydisc.PYDISCDEFS[0]]
    self.cbspec = pydisc.PYDISCSPEC
    self.cbopts = [''] * len(self.cbspec)
    for i in range(len(self.cbspec)): self.cbopts[i] = self.cbspec[i][-1]
    self.cbwids = [[]] * 1
    self.cblbls = [[]] * 1
    colorlbls = ['Scatter']
    for i in range(1):
      self.cbwids[i] = ["combobox", "combobox"]
      self.cblbls[i] = [colorlbls[i] + " (x)", colorlbls[i] + " (y)"]
    self.CB = lbw.LBWidgets(self.Dlg, "Scatter selection", 1, self.cblbls, self.cbwids, self.cbdata, self.cbopts)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Select window", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.Disc1OK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.CB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def Disc1OK(self, ev = None):
    self.UICB = self.CB.retData()
    self.uiID = self.CBx.retData()
    k = -1
    self.discanal = [None] * 2
    self.discspec = [None] * 2
    self.disccens = [None] * 2
    for i in range(1):
      for j in range(2):
        k += 1
        uicb = self.cbspec[self.UICB[i][j]]
        self.discanal[k] = uicb[0]
        self.discspec[k] = uicb[1]
        self.disccens[k] = uicb[2]
    self.DlgClose()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.Lims = self.SetAnal.storeLims([1, 1], self.uiID)
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    if not(self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.Disc1CB)
  def Disc1CB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    uiData = self.extract(self.uiID, win, True)
    self.SetTrig() # implement deletions on self.trig
    _active = self.SetAnal.pw[self.uiID].active # record all current deletions
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.SetDisc = pydisc.pydisc1()
    self.SetDisc.setData(uiData, [self.DataInfo[self.uiID].samplint, 1., 0.], self.onsets, [_active[0], True, _active[2]])
    self.SetDisc.setAnal(self.discanal, self.discspec, self.disccens, si = self.DataInfo[self.uiID].samplint)
    self.SetDisc.setDoneFuncs([self.Disc1Done, self.Disc1Done])
    self.SetAnal.setCursor(self.uiID) # remove cursor
    #self.SetDisc.setForm(form=self.Form.Widget) # this creates a new area and sets it as central Widget
    self.SetDisc.setForm(form=self.Form.Widget, parent=self.Area)       # this preserves the present area
    self.SetGUI(False) # Hiding previous docks  _after_ SetDisc.setForm ensures correct SetDisc dock placement.
    #for _dock in self.SetDisc.Dock: _dock.show() -> no longer needed
    #self.App.Widget.processEvents()              -> superfluous
  def Disc1Done(self, ev = None):
    # self.SetDisc.clrPlots() # pydisc is well behaved and removed all its docks before we reach here
    self.setActive(np.copy(self.SetDisc.active))
    del self.SetDisc
    self.SetDisc = None
    self.SetGUI(True)
    self.SetData(self.overlay, self.Lims, self.active)    
  def Disc3(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Discriminate")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.cbdata = pydisc.PYDISCDEFS
    self.cbspec = pydisc.PYDISCSPEC
    self.cbopts = [''] * len(self.cbspec)
    for i in range(len(self.cbspec)): self.cbopts[i] = self.cbspec[i][-1]
    self.cbwids = [[]] * 3
    self.cblbls = [[]] * 3
    colorlbls = ['Red', 'Green', 'Blue']
    for i in range(3):
      self.cbwids[i] = ["combobox", "combobox"]
      self.cblbls[i] = [colorlbls[i] + " (x)", colorlbls[i] + " (y)"]
    self.CB = lbw.LBWidgets(self.Dlg, "Scatter selection", 1, self.cblbls, self.cbwids, self.cbdata, self.cbopts)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Select window", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.Disc3OK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.CB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def Disc3OK(self, ev = None):
    self.UICB = self.CB.retData()
    self.uiID = self.CBx.retData()
    k = -1
    self.discanal = [None] * 6
    self.discspec = [None] * 6
    self.disccens = [None] * 6
    for i in range(3):
      for j in range(2):
        k += 1
        uicb = self.cbspec[self.UICB[i][j]]
        self.discanal[k] = uicb[0]
        self.discspec[k] = uicb[1]
        self.disccens[k] = uicb[2]
    self.DlgClose()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.Lims = self.SetAnal.storeLims([1, 1], self.uiID)
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    if not(self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.Disc3CB)
  def Disc3CB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    uiData = self.extract(self.uiID, win, True)
    self.SetTrig() # implement deletions on self.trig
    _active = self.SetAnal.pw[self.uiID].active # record all current deletions
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    self.SetDisc = pydisc.pydisc3()
    self.SetDisc.setData(uiData, [self.DataInfo[self.uiID].samplint, 1., 0.], self.onsets, [_active[0], True, _active[2]])
    self.SetDisc.setAnal(self.discanal, self.discspec, self.disccens, si = self.DataInfo[self.uiID].samplint)
    self.SetDisc.setDoneFuncs([self.Disc3Done, self.Disc3Done])
    self.SetAnal.setCursor(self.uiID) # remove cursor
    #self.SetDisc.setForm(form=self.Form.Widget) # this creates a new area and sets it as central Widget
    self.SetDisc.setForm(form=self.Form.Widget, parent=self.Area)       # this preserves the present area
    self.SetGUI(False) # Hiding previous docks  _after_ SetDisc.setForm ensures correct SetDisc dock placement.
    #for _dock in self.SetDisc.Dock: _dock.show() -> no longer needed
    #self.App.Widget.processEvents()              -> superfluous
  def Disc3Done(self, ev = None):
    # self.SetDisc.clrPlots() # pydisc is well behaved and removed all its docks before we reach here
    self.setActive(np.copy(self.SetDisc.active))
    del self.SetDisc
    self.SetDisc = None
    self.SetGUI(True)
    self.SetData(self.overlay, self.Lims, self.active)    
  def ShowTable(self, _uiData = None, _uiSelected = None, _uiOnsets = None):
    self.Lims = self.SetAnal.storeLims([1, 1], self.uiID)
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    uiData = self.extracted[self.active[0], :] if _uiData is None else _uiData
    uiSelected = self.active[1][self.active[0]] if _uiSelected is None else _uiSelected
    uiOnsets = self.onsets[self.active[0]] if _uiOnsets is None else _uiOnsets
    uiOnes = np.ones(len(uiSelected), dtype = bool)
    uiZeros = np.zeros(len(uiSelected), dtype = bool)
    self.SetSumm = pysumm()
    self.SetSumm.setData(uiData, [self.DataInfo[self.uiID].samplint, 1., 0.], uiOnsets, 
        [uiOnes, uiSelected, uiZeros], self.table)
    self.SetSumm.setDoneFuncs([self.TableOK, self.TableCC])
    self.SetSumm.setForm(form=self.Form.Widget, parent=self.Area) 
    self.SetGUI(False)
  def TableOK(self, ev = None):
    pf = self.WriteTable(None, None, self.comment)
    self.TableCC(ev)
    return pf
  def TableCC(self, ev = None):
    #self.SetSumm.clrPlots() # pysumm() is well behaved and clears its own plots
    del self.SetSumm
    self.SetSumm = None
    self.SetGUI(True)
    self.SetData(self.overlay, self.Lims, self.active)    
  def DlgFileSave(self, _opfn = None, _opdn = None, _repl = True):
    opfn, opdn = _opfn, _opdn
    ipfn, ipdn = self.stpf, self.dirn + "/"
    if type(ipfn) is list: ipfn = ipfn[0]
    if opdn is None:
      opdn = ipdn.replace('data', 'analyses') if _repl else ipdn[:]
      if not(os.path.exists(opdn)):
        opdn = ipdn
    if opfn is None: 
      opfn = ipfn + ".tdf"
    senderwidget = lbw.LBWidget(None, None, None, 'base')                        
    return senderwidget.dlgFileSave("Save File", opdn+opfn, '*.tdf', False)
  def WriteTable(self, _opfn = None, _opdn = None, _comm = None):
    pf = self.DlgFileSave(_opfn, _opdn, True)
    if len(pf):
      opdn, opfn = os.path.split(pf)
      self.writetable(opfn, opdn, _comm)
    return pf
  def ExpFit(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Exponential fit")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.Lbx = lbw.LBWidget(self.Dlg, 'Comment (optional): ', 1, 'edit', '')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.ExpFitOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.Lbx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def ExpFitOK(self, event = None):
    self.uiID = self.CBx.retData()
    self._comment = self.Lbx.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID,[0, 0], self.ExpFitCB)
  def ExpFitCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    self.setComment(self._comment)
    self.expfit(self.uiID, win, _pgb)  
    self.ShowTable()
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def PSA(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pstext)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.EDx = lbw.LBWidget(self.Dlg, 'MWD window (s): ', 1, 'edit', '0')
    self.BGB = lbw.BGWidgets(self.Dlg, 'MWD polarity', 0, ["Auto", "Positive", "Negative"], 'radiobutton', 0)
    self.Lbx = lbw.LBWidget(self.Dlg, 'Comment (optional): ', 1, 'edit', '')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.PSAOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.EDx)
    self.Box.add(self.BGB)
    self.Box.add(self.Lbx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def PSAOK(self, event = None):
    self.uiID = self.CBx.retData()
    OK = self.EDx.valiData(1, 1)
    self._comment = self.Lbx.retData()
    if not(OK[0]) or not(OK[1]): return
    self.uiPM = self.BGB.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.edx = abs(float(self.EDx.retData()))
    if self.edx == 0.: 
      self.edx = None
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID,[0, 0], self.PSACB)
  def PSACB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    if self.edx is not None:
      self.edx = abs(0.5 * float(self.edx / self.DataInfo[self.uiID].samplint))
      self.edx = [int(-self.edx), int(self.edx)]
    self.SetActive(self.uiID)       # store any deletions
    polspec = {0:None, 1:False, 2:True}
    _pgb = pgb()
    self.setComment(self._comment)
    self.psa(self.uiID, win, self.edx, polspec[self.uiPM], _pgb)  
    self.ShowTable()
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def MWD(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', 'MWD')
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.EDx = lbw.LBWidget(self.Dlg, 'MWD window (s): ', 1, 'edit', str(4.*self.DataInfo[0].samplint))
    self.Lbx = lbw.LBWidget(self.Dlg, 'Label (optional): ', 1, 'edit', '')
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.MWDOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.EDx)
    self.Box.add(self.Lbx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def MWDOK(self, event = None):
    self.uiID = self.CBx.retData()
    OK = self.EDx.valiData(1, 1)
    self._comment = self.Lbx.retData()
    if not(OK[0]) or not(OK[1]): return
    self.edx = abs(float(self.EDx.retData()))
    if self.edx < self.DataInfo[self.uiID].samplint: return 
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID, [0], self.MWDCB)
  def MWDCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = ev.cursors[0]/self.DataInfo[self.uiID].samplint
    self.SetAnal.setCursor(self.uiID) # remove cursor
    if self.edx is not None:
      self.edx = abs(0.5 * float(self.edx / self.DataInfo[self.uiID].samplint))
      if self.edx > 1: 
        self.edx = [int(win-self.edx), int(win+self.edx)]
      else:
        self.edx = [int(win), int(win+1)]
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    self.setComment(self._comment)
    self.mwd(self.uiID, self.edx)  
    self.ShowTable()
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def Triggers(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Triggers")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    if self.trig is None:
      self.Txt = lbw.LBWidget(self.Dlg, 'It is difficult to perform trigger analysis without triggers.' )
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button"], ["I'll think about it."]) 
      self.BBx.Widgets[0].connect("btndown", self.DlgClose)
    else:  
      self.Txt = lbw.LBWidget(self.Dlg, 'Select trigger window for analysis.' )
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
      self.BBx.Widgets[0].connect("btndown", self.TriggersOK)
      self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.Txt)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def TriggersOK(self, ev = None):
    self.DlgClose()
    self.overlay = self.SetAnal.pw[self.trigID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.trigID].toggleOverlay()
    self.SetAnal.pw[self.trigID].plot.setCursor([0, 0], self.TriggersCB)
  def TriggersCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.setComment()
    self.SetActive(self.trigID)       # store any deletions
    _Res, _Lbl, _Active, _EpiID, _Onsets, _uiData, _I, _ii, _trig = self.triggers(self.trigID, win)  
    self.ShowTable(_uiData, _Active[1], _Onsets)
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def SpikeShape(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Spike Shape Analysis")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.SpikeShapeOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def SpikeShapeOK(self, event = None):
    self.uiID = self.CBx.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.SpikeShapeCB)
  def SpikeShapeCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.setComment()
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    self.spikeShape(self.uiID, win,  _pgb)  
    self.ShowTable()
    self.SetAnal.pw[self.uiID].plot.setCursor() # remove cursor
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def Step(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Current step analysis")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Passsive channel: ', 1, 'combobox', 0, self.chanNames)
    self.BGB = lbw.BGWidgets(self.Dlg, 'Active step type', 0, ["Current step", "Voltage step"], 'radiobutton', self.ivClamp)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.StepOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BGB)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def StepOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.uiIV = self.BGB.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.StepCB)
  def StepCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.setComment()
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    if self.uiIV:
      self.vStep(self.uiID, win,  _pgb)  
    else:
      self.iStep(self.uiID, win,  _pgb)  
    self.ShowTable()
    self.SetAnal.pw[self.uiID].plot.setCursor() # remove cursor
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def IStep(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Current step analysis")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.IStepOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def IStepOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.IStepCB)
  def IStepCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.setComment()
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    self.iStep(self.uiID, win,  _pgb)  
    self.ShowTable()
    self.SetAnal.pw[self.uiID].plot.setCursor() # remove cursor
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def VStep(self, event = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Voltage step analysis")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.VStepOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def VStepOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    self.DlgClose()
    self.SetAnal.setCursor(self.uiID, [0, 0], self.VStepCB)
  def VStepCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.setComment()
    self.SetActive(self.uiID)       # store any deletions
    _pgb = pgb()
    self.vStep(self.uiID, win,  _pgb)  
    self.ShowTable()
    self.SetAnal.pw[self.uiID].plot.setCursor() # remove cursor
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
  def Offset(self, ev = None):
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Offset")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    if not(self.SetAnal.pw[0].active[2].sum()):
      self.CBx = lbw.LBWidget(self.Dlg, 'Offset operation requires at least one selected episode.' )
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button"], ["Oh"]) 
      self.BBx.Widgets[0].connect("btndown", self.DlgClose)
    else:
      self.CBx = lbw.LBWidget(self.Dlg, 'Channel: ', 1, 'combobox', 0, self.chanNames)
      self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["Select window", "Cancel"]) 
      self.BBx.Widgets[0].connect("btndown", self.OffsetOK)
      self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def OffsetOK(self, ev = None):
    self.uiID = self.CBx.retData()
    self.DlgClose()
    self.Active = self.SetAnal.pw[self.uiID].cpyActive()
    self.overlay = self.SetAnal.pw[self.uiID].overlay
    if not (self.overlay):
      self.SetAnal.pw[self.uiID].toggleOverlay()
    xcurs = [0, 0]
    self.SetAnal.setCursor(self.uiID, xcurs, self.OffsetCB)
  def OffsetCB(self, ev = None):  
    if ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    ev.cursors = np.sort(ev.cursors)
    win = [ev.cursors[0], ev.cursors[1]]
    self.SetAnal.setCursor(self.uiID) # remove cursor
    self.Lims = self.SetAnal.storeLims([0, 2], self.uiID)
    self.SetActive(self.uiID)       # store any deletions
    self.offset(self.uiID, None, win)
    self.SetData(self.overlay, self.Lims, self.active)    
  def Export(self, event = None):
    cb0 = ['Do not export offsets', 'Export offset times', 'Export offset indices and sampling header']
    cb1 = ['Do not export traces', 'Export traces as floating point', 'Export raw traces as signed integer with sampling header']
    cb2 = self.chanNames + ["Export entire trace"]
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Export data")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.CB0 = lbw.LBWidget(self.Dlg, 'Export offsets', 1, 'combobox', 2, cb0)
    self.CB1 = lbw.LBWidget(self.Dlg, 'Export traces', 1, 'combobox', 2, cb1)
    self.LBx = lbw.LBWidget(self.Dlg, "Channel selection", 1, "listbox", None, self.chanNames)
    self.LBx.setMode(3, range(self.nchan))
    self.CB2 = lbw.LBWidget(self.Dlg, 'Excerpt cursor selection: ', 1, 'combobox', self.nchan, cb2)
    self.EDx = lbw.LBWidget(self.Dlg, 'Excerpt quantisation (s): ', 1, 'edit', str(self.DataInfo[0].samplint))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.ExportOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.CB0)
    self.Box.add(self.CB1)
    self.Box.add(self.LBx)
    self.Box.add(self.CB2)
    self.Box.add(self.EDx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def ExportOK(self, ev = None):
    OK = self.EDx.valiData(1, 1)
    if not(OK[0]) or not(OK[1]): return
    self.cb0 = self.CB0.retData()
    self.cb1 = self.CB1.retData()
    self.lbx = self.LBx.retData()
    self.cb2 = self.CB2.retData()
    self.edx = float(self.EDx.retData())
    self.DlgClose()
    if not(self.cb1) or not len(self.lbx):
      self.cb1 = 0
      self.lbx = []
    self.Active = self.SetAnal.pw[0].cpyActive()
    self.overlay = self.SetAnal.pw[0].overlay
    if self.cb2 == self.nchan: self.cb2 = None
    if self.cb2 is None or not(self.cb1):
      return self.ExportCB()
    if not(self.overlay):
      self.SetAnal.pw[self.cb2].toggleOverlay()
    xcurs = [0, 0]
    self.SetAnal.setCursor(self.cb2, xcurs, self.ExportCB)
  def ExportCB(self, ev = None):  
    if ev is None:
      win = [0, self.nsamp]
    elif ev.cursors is None: 
      if not(self.overlay): self.SetData(self.overlay) 
      return
    else:
      ev.cursors = np.sort(ev.cursors)
      win = winqran([ev.cursors[0], ev.cursors[1]], self.edx, self.DataInfo[0].samplint, [0, self.nsamp])
      self.SetAnal.setCursor(self.cb2) # remove cursor
    pf = self.DlgFileSave(None, None, False)
    if len(pf):
      self.export(pf, self.cb0, self.cb1, self.lbx, win, np.ravel(self.Active[0]))
  def Select(self, ev = None):
    opttxt = ["Leave active/inactive trace state unchanged", 
              "Activate selected traces leaving others unchanged", "Inactivate selected traces leaving others unchanged",
              "Activate selected and inactivate unselected traces", "Inactivate selected and activate unselected traces"]
    self.active0 = np.copy(self.SetAnal.pw[0].active[0])
    active2 = np.logical_and(self.active0, self.SetAnal.pw[0].active[2])
    actind0 = argtrue(self.active0)
    self.lbt = [''] * len(actind0)
    for i in range(len(actind0)):
      k = actind0[i]
      self.lbt[i] = "".join( ("Trace ", str(k), " (", str(self.onsets[k]*self.SiGnOf[0][0]), ")") )
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', "Select traces")
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBx = lbw.LBWidget(self.Dlg, "Trace selection", 1, "listbox", None, self.lbt)
    self.LBx.setMode(3, list(argtrue(active2)))
    self.CBx = lbw.LBWidget(self.Dlg, 'Selection effects', 1, 'combobox', 0, opttxt)
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button"], ["OK", "Cancel"]) 
    self.BBx.Widgets[0].connect("btndown", self.SelectOK)
    self.BBx.Widgets[1].connect("btndown", self.DlgClose)
    self.Box.add(self.LBx)
    self.Box.add(self.CBx)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()    
  def SelectOK(self, ev = None):
    active2ind = np.array(self.LBx.retData(), dtype = int)
    activefx = self.CBx.retData()
    self.DlgClose()
    active2bool = np.zeros(len(self.lbt), dtype = bool)
    active2bool[active2ind] = True
    active2 = np.zeros(self.nepis, dtype = bool)
    active2[self.active0] = active2bool
    self.active[2][:] = False
    self.active[2][active2] = True
    if activefx == 1:
      self.active[1][self.active[2]] = True
    elif activefx == 2:
      self.active[1][self.active[2]] = False
    elif activefx == 3:
      self.active[1] = np.copy(self.active[2])
    elif activefx == 4:
      self.active[1] = np.logical_not(self.active[2])
    self.active[0] = self.active0
    self.SetActive(self.active)
    for _pw in self.SetAnal.pw:
      _pw.setactive(self.active)
      _pw.setWave()
