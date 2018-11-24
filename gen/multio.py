# multio - a multi input/output multi file multi channel multi format module

import os
from multiprocessing import Process
try:
  import tkFileDialog
except:
  pass
import lbwgui as lbw
import abf
import tdf
import ses
import smr
import si16
import channel
import numpy as np
from lsfunc import *

class IOKeys:
  def __init__(self):
    self.abf = abf.ABF()
    self.smr = smr.SMR()
    self.ses = ses.SES()
    self.tdf = tdf.TDF()
    self.si16 = si16.SI16()
    self.Delimiter = {'.abf': '_', '.smr': '_', '.edr': '_', '.wcp': '_', '.si16': '_', '.tab': '_', '.tdf': '_',
        '.tsv': '_'}
    self.OpenFile = {'.abf': self.abf.OpenFile, '.smr': self.smr.OpenFile, '.edr': self.ses.OpenFile, '.wcp':
        self.ses.OpenFile, '.si16': self.si16.OpenFile, '.tab': self.tdf.readText, '.tdf': self.tdf.readText,
        '.tsv': self.tdf.readText}
    self.ReadChannelInfo = {'.abf': self.abf.ReadChannelInfo, '.smr': self.smr.ReadChannelInfo, '.edr':
        self.ses.ReadChannelInfo, '.wcp': self.ses.ReadChannelInfo, '.si16': self.si16.ReadChannelInfo, '.tab':
        self.tdf.readChan, '.tdf': self.tdf.readChan, '.tsv': self.tdf.readChan}
    self.NumberOfChannels = {'.abf': self.abf.NumberOfChannels, '.smr': self.smr.NumberOfChannels, '.edr':
        self.ses.NumberOfChannels, '.wcp': self.ses.NumberOfChannels, '.si16': self.si16.NumberOfChannels, '.tab':
        self.tdf.ret_nc, '.tdf': self.tdf.ret_nc, '.tsv': self.tdf.ret_nc}
    self.NumberOfEpisodes = {'.abf': self.abf.NumberOfEpisodes, '.smr': self.smr.NumberOfEpisodes, '.edr':
        self.ses.NumberOfEpisodes, '.wcp': self.ses.NumberOfEpisodes, '.si16': self.si16.NumberOfEpisodes, '.tab':
        self.tdf.ret_ne, '.tdf': self.tdf.ret_ne, '.tsv': self.tdf.ret_ne}
    self.NumberOfSamples = {'.abf': self.abf.NumberOfSamples, '.smr': self.smr.NumberOfSamples, '.edr':
        self.ses.NumberOfSamples, '.wcp': self.ses.NumberOfSamples, '.si16': self.si16.NumberOfSamples, '.tab':
        self.tdf.ret_ns, '.tdf': self.tdf.ret_ns, '.tsv': self.tdf.ret_ns}
    self.ReadIntData = {'.abf': self.abf.ReadIntData, '.smr': self.smr.ReadIntData, '.edr': self.ses.ReadIntData,
        '.wcp': self.ses.ReadIntData, '.si16': self.si16.ReadIntData, '.tab': self.tdf.readData, '.tdf':
        self.tdf.readData, '.tsv': self.tdf.readData}
    self.ClearIntData = {'.abf': self.abf.ClearData, '.smr': self.smr.ClearData, '.edr': self.ses.ClearData, '.wcp':
        self.ses.ClearData, '.si16': self.si16.ClearData, '.tab': self.tdf.clearData, '.tdf': self.tdf.clearData,
        '.tsv': self.tdf.clearData}
    self.ReadOnsets = {'.abf': self.abf.ReadOnsets, '.smr': self.smr.ReadOnsets, '.edr': self.ses.ReadOnsets, '.wcp':
        self.ses.ReadOnsets, '.si16': self.si16.ReadOnsets, '.tab': self.tdf.readOnsets, '.tdf': self.tdf.readOnsets,
        '.tsv': self.tdf.readOnsets}
    self.CloseFile = {'.abf': self.abf.CloseFile, '.smr': self.smr.CloseFile, '.edr': self.ses.CloseFile, '.wcp':
        self.ses.CloseFile, '.si16': self.si16.CloseFile, '.tab': self.tdf.initialise, '.tdf': self.tdf.initialise,
        '.tsv': self.tdf.initialise}

class filio:
  ioks = IOKeys()
  path = None
  dirn = None
  filn = None
  stem = None
  extn = None
  data = None
  stpf = None # stem prefix
  stsf = None # stem suffix
  dstr = None # stem delimiting string
  defextn = '.si16'
  maxi = 10000000 # maximum number of read sample points for interpreting channels
  def initialise(self, _path = None, _options = None):
    self.data = []
    self.open(_path, _options)
  def open(self, _path = None, _options = None, _nofilter = 0, _senderwidget = None): # last 2 inputs argument necessary for inheriting class
    if _path == '': return # returns function call from constructor
    elif _path == None:
        senderwidget = _senderwidget
        if senderwidget == None:
          senderwidget = lbw.LBWidget(None, None, None, 'base')
        self.path = senderwidget.dlgFileOpen("Open Data File")
        i = -1
        if self.path == "":
          self.path = None
        else:
          i = self.path.find("'")
        if i == 2: # remove string elements comprising apostrophes, `u' characters, etc...
          self.path = self.path[i+1:]
          i = self.path.find("'")
          self.path = self.path[:i]
    else:
        self.path = _path
    if self.path == None: return None
    self.parse(None, _nofilter)
    return self.path
  def parse(self, _path = None, _nofilter = 0): # last input argument necessary for inheriting class
    if _path != None: self.path = _path
    self.parsePath()
  def parsePath(self):
    if self.path == None: return
    self.dirn, self.filn = os.path.split(self.path)
    self.stem, self.extn = os.path.splitext(self.filn)
    self.extn = self.extn.lower()
    # test it
    try:
      _ = self.ioks.Delimiter[self.extn]
    except KeyError: # default it to something innocuous
      self.extn = self.defextn
    self.parseStem()
  def parseStem(self):
    if not(self.ioks.Delimiter.has_key(self.extn)): # If file extension not recognised
      self.dstr = None
      self.stpf = None
      self.stsd = None
      return
    self.dstr = self.ioks.Delimiter[self.extn]
    parsedstem = self.stem.split(self.dstr)
    if len(parsedstem) == 1:
      self.stpf = parsedstem[0]
      self.stdf = None
      return
    elif len(parsedstem) == 2:
      self.stpf = parsedstem[0]
      self.stsf = parsedstem[1]
      return
    else:
      self.stsf = parsedstem[-1]
      self.stpf = self.stem[0:(len(self.stem)-len(self.stsf)-len(self.dstr))]
  def readchan(self):
    self.ioks.OpenFile[self.extn](self.path)
    self.chan = self.ioks.ReadChannelInfo[self.extn]()
    self.nc = len(self.chan)
    self.ioks.CloseFile[self.extn]()
    self.chit = []
    self.chtl = []
    self.chis = []
    for i in range(self.nc):
      self.chis.append("Ch" + str(self.chan[i].index))
      self.chtl.append(self.chan[i].name)
      self.chit.append("Ch" + str(self.chan[i].index) + " (" + self.chan[i].name + ")")
  def readddim(self): # read dimensions
    self.ioks.OpenFile[self.extn](self.path)
    self.nc = self.ioks.NumberOfChannels[self.extn]()
    self.ne = self.ioks.NumberOfEpisodes[self.extn]()
    self.ns = self.ioks.NumberOfSamples[self.extn]()
    self.ioks.CloseFile[self.extn]()
  def readdata(self, _path = None):
    self.open(_path , None, 1) # last argument circumvents filtering but parsing is enabled
    self.ioks.OpenFile[self.extn](self.path)
    self.chan = self.ioks.ReadChannelInfo[self.extn]()
    self.nc = self.ioks.NumberOfChannels[self.extn]()
    self.ne = self.ioks.NumberOfEpisodes[self.extn]()
    self.ns = self.ioks.NumberOfSamples[self.extn]()
    self.data = self.ioks.ReadIntData[self.extn]()
    self.onsets = self.ioks.ReadOnsets[self.extn]()
    self.ioks.CloseFile[self.extn]()
  def cleardata(self):
    if self.data is not None:
      del self.data
      self.data = None
    self.ioks.ClearIntData[self.extn]()
  def interpret(self, nsamp = 1024): # constructs list indexing likely passive vs active channel pairs
    self.readddim() # read dimensions
    self.readchan() # read channel info
    nsamp = min(nsamp, self.ns)
    self.ncp = (self.nc) / 2
    self.ncp += not(self.ncp)
    if self.nc == 1: # handle exception first
      self.chpa = [[0, 0]]
      self.ctpa = ["Pair 0"]
      self.chtl = self.chan[0].name
      return
    if self.nc * self.ns * self.ne <= self.maxi: # only open if not a huge file
      self.readdata(self.path)
      s = np.empty( (self.nc), dtype = float)
      for i in range(self.nc): # extract info from first episode only
        s[i] = np.std(self.data[i][0][:nsamp]) * float(self.chan[i].gain)
    else:
      s = np.arange(self.nc)
    i = np.argsort(s)
    ia = i[:self.ncp]
    ip = i[-self.ncp:]
    iia = np.argsort(ia)
    ip = ip[iia]
    self.chpa = []
    self.ctpa = []
    for i in range(self.ncp):
      self.ctpa.append(''.join(("Pair ", str(i))))
      self.chpa.append( [ip[i], ia[i]] )
  def save(self, _path = None, _options = None, _nofilter = 0, _senderwidget = None): # last 2 inputs argument necessary for inheriting class
    if _path == '': return # returns function call from constructor
    elif _path == None:
        senderwidget = _senderwidget
        if senderwidget == None:
          senderwidget = lbw.LBWidget(None, None, None, 'base')
        self.path = senderwidget.dlgFileSave("Save File")
        if self.path == "": self.path = None
    else:
        self.path = _path
    if self.path == None: return None
    return self.path

class mfilio (filio):
  relf = None # related file list (common stem prefices)
  cesi = [] # common episode-duration indices (indexing relf)
  cefn = [] # common episode-duration filenames
  seli = None # selected index from cefn
  ed = None # episduration
  chin = [] # channel indices to order channels when inputting data
  comf = filio() # a class instantation for comparing for common file stems etc...
  Data = None
  def setfiles(self, _stpf, _stsflist, _extn): # a non_GUI override function
    self.extn = _extn
    self.cefn = []
    for i in length(_stsflist):
      self.cefn.append(os.path.join( _stpf, _stsf, _extn ) )
  def parse(self, _path = None, _nofilter = 0):
    if _path != None: self.path = _path
    self.parsePath()
    if not(_nofilter):
      self.filterRelatedFiles()
      self.filterCommonEpisodeDurationFiles()
  def filterRelatedFiles(self):
    if self.stpf == None:
      self.relf = [self.filn]
    else:
      dirList = os.listdir(self.dirn)
      self.relf = []
      for fn in dirList:
        if fn == self.filn:
          self.relf.append(fn)
        else:
          self.comf.parse(fn)
          if self.stpf == self.comf.stpf and self.extn == self.comf.extn:
            self.relf.append(fn)
  def filterCommonEpisodeDurationFiles(self):
    if self.relf == None: return
    self.readddim()
    self.cesi = []
    self.cefn = []
    count = 0
    for i in range(len(self.relf)):
      fn = self.relf[i]
      try:
        self.ioks.OpenFile[self.extn](os.path.join(self.dirn, fn))
        if self.nc == self.ioks.NumberOfChannels[self.extn]() and self.ns == self.ioks.NumberOfSamples[self.extn]():
          self.cesi.append(i)
          self.cefn.append(fn)
          if fn == self.filn:
            self.seli = count
          else:
            count += 1
      except:
        pass
      self.ioks.CloseFile[self.extn]()
  def readData(self, filn = None, chan = None, _dirn = None):
    if (filn == None): filn = self.cefn
    if (chan == None): chan = range(self.nc)
    if (_dirn == None): _dirn = self.dirn
    nfn = len(filn)
    nch = len(chan)
    self.Data = listmat(nch, nfn)
    self.Onsets = [[]] * nfn
    lastonset = None
    for j in range(nfn):
      self.readdata(os.path.join(_dirn, filn[j]))
      if not(j):
        self.Onsets[j] = np.copy(self.onsets)
      else:
        self.Onsets[j] = np.copy(self.onsets) + (self.Onsets[j-1][-1] + self.ns)
      for i in range(nch):
        self.Data[i][j] = self.data[chan[i]] # this copies by data, not address
      self.cleardata()
    self.Onsets = np.hstack(self.Onsets)
  def clear(self):
    if self.Data is None: return
    del self.Data
    self.Data = None

class mfilioGUI(mfilio):
  lbData = None
  cbData = None
  etData = None
  OK = 0
  def Open(self, okfunc, ccfunc, _path = None):
    self.form = lbw.LBWidget(None, None, None, 'dlgform', "File and channel selection")
    self.open(_path, None, 0, self.form)
    return self.OpenDlg(okfunc, ccfunc)
  def OpenDlg(self, okfunc, ccfunc):
    if self.seli == None or len(self.cefn) == 0: return None
    self.interpret() # to initialise channel list
    self.box = lbw.LBWidget(self.form, None, 1)
    self.lb = lbw.LBWidget(self.form, "File Selection", 1,"listbox", self.seli, self.cefn)
    self.lb.setMode(4)
    self.box.add(self.lb)
    self.cbwids = []
    self.cblbls = []
    self.cbdata = []
    self.cbopts = []
    chitoff = self.chit
    chitoff.append("Off")
    for i in range(self.ncp):
      self.cbwids.append(["combobox", "combobox"])
      self.cblbls.append(["Passive Signal #" + str(i), "Active Signal #" + str(i)])
      if i == 0: # included this branch code to turn `off' all other channels
        self.cbdata.append(self.chpa[i])
      else:
        self.cbdata.append([self.nc, self.nc])
      self.cbopts.append([chitoff, chitoff])
    self.cb = lbw.LBWidgets(self.form, "Channel Selection", 1, self.cblbls, self.cbwids, self.cbdata, self.cbopts)
    self.box.add(self.cb)
    self.lewids = []
    self.lelbls = []
    self.ledata = []
    for i in range(self.nc):
      self.lewids.append(["edit", "edit", "edit"])
      self.lelbls.append([self.chis[i] + " samp. int. (s)", self.chis[i] + " gain", self.chis[i] + " offset"])
      self.ledata.append([str(self.chan[i].samplint), str(self.chan[i].gain), str(self.chan[i].offset)])
    self.et = lbw.LBWidgets(self.form, "Channel Information", 1, self.lelbls, self.lewids, self.ledata)
    self.box.add(self.et)
    self.bbx = lbw.BWidgets(self.form, 0, None, ["Button", "Button"], ["OK", "Cancel"])
    self.box.add(self.bbx)
    self.bbx.Widgets[0].connect("btndown", okfunc)
    self.bbx.Widgets[1].connect("btndown", ccfunc)
    self.form.setChild(self.box)
    self.form.show()
  def CloseDlg(self, _OK = 0):
    self.OK = _OK
    if not(self.OK):
      self.form.close()
      return None
    self.lbData = self.lb.retData()
    if not(len(self.lbData)):
      self.mbox = lbw.LBWidget(None, None, None, 'dlgmess', 'No file selected.')
      self.mbox.show()
      return None
    self.cbData = self.cb.retData()
    for cbdata in self.cbData:
      if (cbdata[0] == self.nc and cbdata[1] != self.nc) or (cbdata[0] != self.nc and cbdata[1] == self.nc):
        self.mbox = lbw.LBWidget(None, None, None, 'dlgmess', 'Full passive and active channel pairs must be selected.')
        self.mbox.show()
        return None
    validet, discard = (self.et.valiData(1,1))
    if not(validet):
      self.mbox = lbw.LBWidget(None, None, None, 'dlgmess', 'Invalid channel information provided.')
      self.mbox.show()
      return None
    else:
      self.etData = self.et.retData()
      self.form.close()
    filn = []
    for i in range(len(self.lbData)): filn.append(self.cefn[self.lbData[i]])
    chno = []
    signof = []
    chans = []
    for i in range(len(self.cbData)):
      j = i * 2
      k = j + 1
      if self.cbData[i][0] < self.nc and self.cbData[i][1] < self.nc:
        chp = self.cbData[i][0]
        cha = self.cbData[i][1]
        chno.append(chp)
        chno.append(cha)
        signof.append([float(self.etData[chp][0]), float(self.etData[chp][1]), float(self.etData[chp][2])])
        signof.append([float(self.etData[cha][0]), float(self.etData[cha][1]), float(self.etData[cha][2])])
        chans.append(channel.chWave())
        chans[j].index = self.chan[chp].index
        chans[j].name = self.chan[chp].name
        chans[j].units = self.chan[chp].units
        chans[j].samplint = signof[j][0]
        chans[j].gain = signof[j][1]
        chans[j].offset = signof[j][2]
        chans.append(channel.chWave())
        chans[k].index = self.chan[cha].index
        chans[k].name = self.chan[cha].name
        chans[k].units = self.chan[cha].units
        chans[k].samplint = signof[k][0]
        chans[k].gain = signof[k][1]
        chans[k].offset = signof[k][2]
    return self.dirn, filn, chno, signof, chans

