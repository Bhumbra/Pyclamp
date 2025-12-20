import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyclamp.gui.pyqtplot as pq
import pyclamp.gui.pyplot as pyplot
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.fpanal import *
from pyclamp.gui.xyfunc import *
from pyclamp.gui.pyplot import *
from pyclamp.gui.pgb import pgb
from pyclamp.dsp.channel import chWave
QtApply = QtWidgets.QDialogButtonBox.StandardButton.Apply
QtCancel = QtWidgets.QDialogButtonBox.StandardButton.Cancel

class pywavescat: # combines a wave plot with a scatter plot with a discriminating ellipse
  form = None
  area = None
  gbox = None
  parent = None
  pick = None
  wave = None
  scat = None
  active = None
  useDocks = False
  on = None
  wact1 = None
  def __init__(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], _X = [], _Y = [], _xy=None):
    self.setData(_data, _chinfo, _lat, _Active, _X, _Y, _xy)
  def setData(self, _data = [], _chinfo = [], _lat = [], _Active = [True, True, False], _X = [], _Y = [], _xy=None):
    self.iniPick(_data, _chinfo, _lat, [_Active[0], _Active[2], False])
    self.iniWave(_data, _chinfo, _lat, _Active)
    self.iniScat(_X, _Y, _xy)
  def iniPick(self, _data = [], _chinfo = [], _lat = [], _Active = [True, False, False]):
    if len(_data):
      self.pick = pywave(_data, _chinfo, _lat, _Active)
      self.pick.setShowInactive(False)
      self.pick.setPen(None, None, None) # picker can't pick
  def iniWave(self, _data = [], _chinfo = [], _lat = [], _Active = [True, True, False]):
    self.data = _data
    self.Active = _Active
    if len(_data):
      self.wave = pywave(self.data, _chinfo, _lat, self.Active)
      self.active = self.wave.active
  def iniScat(self, _X = [], _Y = [], _xy=None):
    if len(_X):
      if self.active is None:
        raise ValueError("Wave data must be initialised before scatter data.")
    self.scat = pyscat(_X, _Y, self.active)
    if _xy is None: _xy = ['x', 'y']
    self.scat.setLabels(_xy)
  def setArea(self, *args, **_kwds):   
    self.newparent = False
    kwds = dict(_kwds)
    if 'form' in kwds:
      self.form = kwds['form']
      del kwds['form']
    else:
      pass
    if 'parent' in kwds:
      self.parent = kwds['parent']
      if type(self.parent) is BaseAreaClass:
        self.area = self.parent()
      elif type(self.parent) is area:
        self.area = self.parent
      elif type(self.parent) is BaseGboxClass:
        self.gbox = self.parent
      if self.form is None:
        self.form = self.parent.parent()
        while type(self.form) is not BaseFormClass:
          self.form = self.form.parent()
    else:
      if self.form is None: self.form = BaseFormClass()
    if self.parent is None:
      self.newparent = True
      self.area = area()
      self.parent = self.area
    if self.form is not None:  
      self.form.setCentralWidget(self.parent)  
    self.useDocks = self.area is not None
    return self.parent
  def setPlots(self, I = range(3), *args, **kwds):
    if self.area is None: self.setArea(*args, **kwds)
    if not(self.useDocks): 
      if listfind(I, 0) is not None:
        self.pick.setPlot(parent = self.gbox, row = 1)
        self.pick.setActiveChangedFunc(self.onActiveChanged)
        self.pick.toggleOverlay()
      if listfind(I, 1) is not None:  
        self.wave.setPlot(parent = self.gbox, row = 1)
        self.wave.setActiveChangedFunc(self.onActiveChanged)
        self.wave.toggleOverlay()
      if listfind(I, 2) is not None:
        self.gbox.nextRow()
        self.scat.setPlot(parent = self.gbox, row = 0)
        self.scat.setActiveChangedFunc(self.onActiveChanged)
    else:
      self.docks = [None] * 3 # pick, wave, scat
      self.boxes = [None] * 3 # pick, wave, scat
      dockTitles = ["Channel", "Channel", "Scatter"]
      if isinstance(self.wave.chinfo, chWave):
        dockTitles[0] = self.wave.chinfo.name
        dockTitles[1] = self.wave.chinfo.name
      pickwavescat = [self.pick, self.wave, self.scat]
      for i in I:
        dtit = dockTitles[i] if i < 2 else None
        self.docks[i] = dock(dtit)
        self.boxes[i] = gbox()
        self.docks[i].add(self.boxes[i])
        if i == 0:
          self.area.add(self.docks[i], 'bottom')
        elif i == 1:
          self.area.add(self.docks[i], 'right', self.docks[i-1])
        else:
          self.area.add(self.docks[i], 'top')
        pickwavescat[i].setPlot(parent = self.boxes[i])
        pickwavescat[i].setActiveChangedFunc(self.onActiveChanged)
        if (i < 2): pickwavescat[i].toggleOverlay()
    return self.parent
  def clrPlots(self, keepParent = False):
    if not(self.useDocks):
      if self.pick is not None:
        self.gbox.remove(self.pick.plot)
        del self.pick; 
        self.pick = None
      if self.wave is not None:
        self.gbox.remove(self.wave.plot)
        del self.wave; 
        self.wave = None
      if self.scat is not None:
        self.gbox.remove(self.scat.plot)
        del self.scat; 
        self.scat = None
    self.area.clrDocks(self.docks)
    self.docks = None
    self.boxes = None
    if not(self.newparent) or keepParent or self.area is None: return
    self.area.remove()
    del self.area
    self.area = None
  def setForm(self, *args, **kwds):
    self.setPlots(**kwds)
    self.setScat(*args)
  def discriminate(self):
    if self.on is None: return
    self.on = np.logical_and(self.wave.active[0], self.on)
  def setScat(self, spec = 'm'):
    if self.on is None:
      self.scat.setEllipse(spec)
    self.scat.setScat()
    self.on = np.logical_and(self.wave.active[0], self.scat.active[1])
  def setWaves(self, spec = 0):
    if spec == 2 and self.wact1 is not None: # check we can get away with only showing picks
      if not(np.all(self.wave.active[2] == self.wact1)):
        spec = 0 # no we can't
    waveactive = np.logical_or(self.on, self.wave.active[2])
    self.wave.setactive([None, waveactive, None])
    self.pick.setPens(self.wave.pens)
    if spec == 2:
      self.wave.setPick()
    else:
      self.wave.setWave()
      self.wact1 = np.copy(self.wave.active[1])
    self.pick.setWave() 
  def onActiveChanged(self, ev):
    if ev.action != 2: # discriminate if necessary
      self.discriminate()
      if ev.sender == self.scat:                # we need to set the scatters to work out the new colours
        self.setScat(ev.action)
    if ev.sender != self.scat:                            # i.e. event occurred outside scatters
      if ev.sender == self.pick:             # inside pick
        if ev.action == 2:
          self.pick.setActive([None, self.pick.Active[2], False]) 
          self.wave.setActive([None, None, self.pick.Active[1]]) 
          self.scat.setActive([None, None, self.pick.active[1]])
        elif ev.action == 0: # a master chooses, a slave obeys    
          self.wave.keyPressEvent(ev)
        else:
          print("Warning: Unknown call from self.pick") # should have no discrimination
      elif ev.sender == self.wave:           # inside wave
        if ev.action == 2:                   # just picking
          self.pick.setActive([None, self.wave.active[2], None])
          self.scat.setActive([None, None, self.wave.active[2]])
        elif ev.action == 0: # self.wave is the master; the rest are slaves
          self.active = self.wave.active
          self.pick.setActive([self.wave.active[0], None, None]) 
          self.scat.setActive([self.wave.active[0], None, None])
        else:  
          pass # just discriminating
        self.active = np.copy(self.wave.active)
      elif ev.sender == self:    # first plot
        pass # we have already setScat by this stage
      else:
        print("Unknown sender ID: " + str(ev.sender))
    else:                    # insider scatter
      if ev.action == 2:  # just picking
        self.pick.setactive([None, self.scat.active[2], None])
        self.wave.setactive([None, None, self.scat.active[2]])
      elif ev.action == 0:   # delete/insert
        self.wave.keyPressEvent(ev)
        self.active = self.wave.active # register deletions/insertions
      else:  
        self.active[1] = np.copy(self.scat.active[1])  # just discriminating
    self.setWaves(ev.action) 
    if ev.action == 2 or ev.sender != self.scat: self.setScat(ev.action)
    return ev.sender

class pyscat3: # combines three scatter plots for discrimination.
  form = None
  area = None
  gbox = None
  Bbox = None
  Dock = None
  parent = None
  defEllCols = [[1,0,0], [0,1,0], [0,0,1]]
  defMkrSize = [2, 3, 5] # default [off, on, pick] marker sizes
  defColMode = 0 # 0 = monochrome, 1 = logical, 2 = ordinal
  defLogCols = [[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]]
  defUseBool = [ True,   True,   True,   True,   False,  False,  False,  False]
  defLogical = [[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
  useDocks = False
  btnwidth = 50
  btnWidth = 100
  colco = [63., 255.]
  colgr = 0.25
  scats = None
  pens = []
  brushes = []
  sizes = []
  logic3 = None
  logic8 = None
  colind = None
  _sizes = None
  pickcolind = -1
  def __init__(self, _XXX = None, _YYY = None, _active = [True, True, False], _xy3 = [None, None, None]):
    self.setDefs()
    self.setData(_XXX, _YYY, _active, _xy3)
  def setDefs(self):
    if pyplot.DEFSCATMARKERSIZE is not None:
      self.defMkrSize[1] = pyplot.DEFSCATMARKERSIZE
      self.defMkrSize[0] = int(round(2.*self.defMkrSize[1]/3.))
      self.defMkrSize[2] = int(round(5.*self.defMkrSize[1]/3.))
    self.setDoneFuncs()
    self.setEllCols(self.defEllCols, pyplot.JetRGBC)
    self.setMkrSize(self.defMkrSize)
    self.setColMode(self.defColMode)
    self.setLogCols(self.defLogCols)
    self.setUseBool(self.defUseBool)
    self.setLogical(self.defLogical)
  def setEllCols(self, _ellCols = None, ellColc = None): 
    if _ellCols is not None: self.ellCols = _ellCols  
    if ellColc is None: return
    for i in range(len(self.ellCols)):
      self.ellCols[i] = list(np.minimum(np.array(self.ellCols[i], dtype = float) * ellColc, 1.))
  def setMkrSize(self, _mkrSize = None): 
    if _mkrSize is not None: self.mkrSize = _mkrSize 
  def setColMode(self, _colMode = None):
    if _colMode is not None: self.colMode = _colMode  
  def setLogCols(self, _logCols = None):
    if _logCols is not None: self.logCols = _logCols
    if type(self.logCols is list):
      self.logCols = np.array(self.logCols, dtype = float)
    self.pen8 = [None] * 8
    self.brush8 = [None] * 8
    c = None if pyplot.JetRGBC is None else 255. * pyplot.JetRGBC
    for i in range(8):
      self.pen8[i] = rgbPen(self.logCols[i], c)
      self.brush8[i] = rgbBrush(self.logCols[i], c)
    self.counts = np.empty(8, dtype = int)
  def setUseBool(self, _useBool = None):
    if _useBool is not None: self.useBool = _useBool
  def setLogical(self, _logical = None):
    if _logical is not None: self.logical = _logical   
  def setData(self, _XXX = None, _YYY = None, _active = [True, True, False], _xy3 = [None, None, None]):
    self.iniScats(_XXX, _YYY, _active, _xy3)
  def iniScats(self, _XXX = None, _YYY = None,_active = [True, True, False], _xy3 = [None, None, None]):
    if _XXX is None: return
    self.n = 0
    try:
      self.n = len(_XXX[0])
    except IndexError: # not ready yet
      return
    self.scats = [None] * 3
    for i in range(3):
      _xy = _xy3[i]
      if _xy is None: _xy = ['x', 'y']
      self.scats[i] = pyscat(_XXX[i], _YYY[i], _active)
      self.scats[i].setLabels(_xy)
      if len(_XXX[i]) != self.n or len(_YYY[i]) != self.n:
        raise ValueError("Mismatches in data sizes")
  def setArea(self, *args, **_kwds):
    kwds = dict(_kwds)
    if 'form' in kwds:
      self.form = kwds['form']
      del kwds['form']
    else:
      pass
    self.newparent = False
    if 'parent' in kwds:
      self.parent = kwds['parent']
      if type(self.parent) is BaseAreaClass:
        self.area = self.parent()
      elif type(self.parent) is area:
        self.area = self.parent
      elif type(self.parent) is BaseGboxClass:
        self.gbox = self.parent
      self.form = self.parent.parent()
      while type(self.form) is not BaseFormClass:
        self.form = self.form.parent()
    else:
      if self.form is None: self.form = BaseFormClass()
    if self.parent is None:
      self.newparent = True
      self.area = area()
      self.parent = self.area
    if self.form is not None:  
      self.form.setCentralWidget(self.parent)  
    self.useDocks = self.area is not None
    return self.parent
  def setPlots(self, *args, **kwds):
    if self.scats is None:
      raise ValueError("Cannot set plots without data.")
    if self.area is None: self.setArea(*args, **kwds)
    if not(self.useDocks):
      for i in range(3):
        self.scats[i].setPlot(parent = self.gbox, col = i, *args, **kwds)
        self.scats[i].setActiveChangedFunc(self.onActiveChanged)
        self.scats[i].setEllipse(self.ellCols[i])
        self.scats[i].setSelectFuncs(None, None, None) # Override select
        self.scats[i].setPickFuncs(amplify, None, self.retPickSize) # Override pick
    else:
      self.Docks = [None] * 3
      self.Boxes = [None] * 3
      for i in range(3):
        self.Docks[i] = dock('') # quotes here stop treatment of docks as 2nd-class citizens
        self.Boxes[i] = self.Docks[i].addGbox()
        if i:
          self.area.add(self.Docks[i], 'right', self.Docks[i-1])
        else:
          self.area.add(self.Docks[i])
        self.scats[i].setPlot(parent = self.Boxes[i])
        self.scats[i].setActiveChangedFunc(self.onActiveChanged)
        self.scats[i].setEllipse(self.ellCols[i])
        self.scats[i].setSelectFuncs(None, None, None) # Override select
        self.scats[i].setPickFuncs(amplify, None, self.retPickSize) # Override pick
    return self.parent
  def clrPlots(self, keepParent = False):
    if not(self.useDocks):
      for i in range(3):
        self.gbox.remove(self.scats[i].plot)
      return
    self.area.clrDocks(self.Dock)
    self.area.clrDocks(self.Docks)
    self.Docks = None
    self.Boxes = None
    self.Dock = None
    self.Bbox = None
    if not(self.newparent) or keepParent or self.area is None: return
    self.area.remove()
    del self.area
    self.area = None
  def iniBtns(self, I = range(2)):
    if not(self.useDocks): return
    self.Bbox = [None] * 2
    self.Dock = dock()
    self.area.add(self.Dock, 'bottom')
    for i in I:
      self.Bbox[i] = self.Dock.addBbox()
      if i == 0:
        for j in range(8):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnwidth)
          self.Bbox[i].setIconSize(j, QtCore.QSize(1,1))
        self.Bbox[i].Connect(0, self.toggleAccept0)
        self.Bbox[i].Connect(1, self.toggleAccept1)
        self.Bbox[i].Connect(2, self.toggleAccept2)
        self.Bbox[i].Connect(3, self.toggleAccept3)
        self.Bbox[i].Connect(4, self.toggleAccept4)
        self.Bbox[i].Connect(5, self.toggleAccept5)
        self.Bbox[i].Connect(6, self.toggleAccept6)
        self.Bbox[i].Connect(7, self.toggleAccept7)
      else:
        self.Bbox[i].addButton(QtCancel)
        self.Bbox[i].addButton(QtApply)
        self.Bbox[i].setWidth(0, self.btnWidth)
        self.Bbox[i].setWidth(1, self.btnWidth)
        self.Bbox[i].Connect(0, self.cancelClick)
        self.Bbox[i].Connect(1, self.applyClick)
  def setForm(self, **kwds):
    self.setPlots(**kwds)
    self.iniBtns()
    self.discriminate()  
    self.setScats()
    self.setBtns()
  def setDoneFuncs(self, _doneFuncs = None):
    if not(isarray(_doneFuncs)): _doneFuncs = [_doneFuncs]
    if len(_doneFuncs) == 1: _doneFuncs = [_doneFuncs[0], None]
    self.doneFuncs = _doneFuncs
  def applyClick(self):
    if self.doneFuncs[0] is None:
      self.clrPlots()
      print("Apply Button Clicked.")
      return
    self.doneFuncs[0]()
  def cancelClick(self):
    if self.doneFuncs[0] is None:
      self.clrPlots()
      print("Cancel Button Clicked.")
      return
    self.doneFuncs[1]()
  def retPickSize(self, _active = True):
    if _active: return self.mkrSize[2]
    return 0
  def setBtns(self, spec = 0, ind = 0):
    if spec == 2: return
    palroles = [QtGui.QPalette.Inactive, QtGui.QPalette.Active]
    txtroles = ['Off', 'On']
    for i in range(8):
      hicol = np.array(self.logCols[i], dtype = float) * self.colco[int(self.useBool[i])]
      locol = hicol * self.colgr
      self.Bbox[ind].setCols(i, hicol, locol)
      self.Bbox[ind].setText(i, txtroles[int(self.useBool[i])])
    self.setApply()  
  def setApply(self, applytxt = None):
    if applytxt is None: applytxt = str(len(np.nonzero(self.on)[0]))
    self.Bbox[-1].setText(1, applytxt)  
  def discriminate(self):
    if self.logic3 is None: self.logic3 = np.empty((3, self.n), dtype = float)
    if self.logic8 is None: self.logic8 = np.empty((8, self.n), dtype = float)
    if self.colind is None: self.colind = np.empty(self.n, dtype = int)
    self.On = [self.scats[0].active[1], self.scats[1].active[1], self.scats[2].active[1]]
    OfOn = [np.logical_not(self.On), self.On]  
    self.on = np.zeros(self.n, dtype = bool)
    for i in range(8):
      for j in range(3):
        self.logic3[j] = OfOn[self.logical[i][j]][j]
      self.logic8[i] = np.logical_and(self.logic3[0], np.logical_and(self.logic3[1], self.logic3[2]))
      k = np.nonzero(self.logic8[i])[0]
      self.counts[i] = len(k)
      self.colind[k] = i
      if self.useBool[i]: self.on[k] = True
  def setScats(self, spec =  0, I = range(3)): # we also handle sizes here. #spec = 0:both, 1:scats only, 2:picks only
    if not(len(self.pens)): self.pens = [None] * self.n
    if not(len(self.brushes)): self.brushes = [None] * self.n
    self.active = self.scats[0].active
    self.sizes = np.tile(self.mkrSize[0], self.n)
    self.sizes[self.on] = self.mkrSize[1]
    self.sizes[self.active[2]] = self.mkrSize[1]
    if spec == 2: # if supposedly only picking
      if not(np.all(self._sizes == self.sizes)):
        spec = 0
    self._sizes = np.copy(self.sizes)
    if spec != 2: # if not only picking
      for i in range(self.n):
        if not(self.active[0][i]):
          self.pens[i] = nullpen()
          self.brushes[i] = nullbrush()
        else:
          self.pens[i] = self.pen8[self.colind[i]]
          self.brushes[i] = self.brush8[self.colind[i]]
      for i in I:
        self.scats[i].setPen(pen=self.pens,update=True)
        self.scats[i].setBrush(brush=self.brushes,update=True)
        self.scats[i].setSize(size=self.sizes, update=True)
    _pickcolind = self.colind[self.scats[0].active[2]] 
    if spec == 1: # if picks are supposedly unaffected
      if not(np.all(_pickcolind == self.pickcolind)): # if not matching sizes, don't hide such an error here
        spec = 0
    self.pickcolind = _pickcolind
    if spec != 1:
      for i in I:
        self.scats[i].setPick()
  def toggleAccept0(self, ev = None):
    return self.toggleAccept(0)
  def toggleAccept1(self, ev = None):
    return self.toggleAccept(1)
  def toggleAccept2(self, ev = None):
    return self.toggleAccept(2)
  def toggleAccept3(self, ev = None):
    return self.toggleAccept(3)
  def toggleAccept4(self, ev = None):
    return self.toggleAccept(4)
  def toggleAccept5(self, ev = None):
    return self.toggleAccept(5)
  def toggleAccept6(self, ev = None):
    return self.toggleAccept(6)
  def toggleAccept7(self, ev = None):
    return self.toggleAccept(7)
  def toggleAccept(self, i):
    self.useBool[i] = not(self.useBool[i])
    self.discriminate()
    self.setScats(1) # doesn't effect previews
    self.setBtns(1) 
  def onActiveChanged(self, ev):
    k = None
    if ev.sender == self.scats[0]: k = 0
    if ev.sender == self.scats[1]: k = 1
    if ev.sender == self.scats[2]: k = 2
    if k is None: return k
    if ev.action == 2: # just picking
      I = []
      for i in range(3):
        if i != k:
          self.scats[i].active[2] = np.copy(self.scats[k].active[2])
          I.append(i)
      self.setScats(ev.action, I)
      return k 
    self.discriminate()
    self.setScats(ev.action) # just discriminating
    self.setBtns(ev.action)
    return k

class pywavescat3 (pywavescat, pyscat3): # combines a wave plot with a 3 scatter plot each with a discrimnating ellipses
  Bbox = None
  Dock = None
  wact1 = None
  def __init__(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], _XXX = None, _YYY = None,
      _xy3 = [None, None, None]):
    self.setDefs()
    self.setData(_data, _chinfo, _lat, _Active, _XXX, _YYY, _xy3)
  def setData(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], _XXX = None, _YYY = None, _xy3 = [None, None, None]):
    pywavescat.setData(self, _data, _chinfo, _lat, _Active, [], [])
    pyscat3.setData(self, _XXX, _YYY, _Active, _xy3)
    if self.wave is None: return
    self.activeorig = np.copy(self.wave.active) 
  def setForm(self, **kwds):
    pyscat3.setPlots(self, **kwds)
    pywavescat.setPlots(self, range(2), **kwds)
    self.wave.setShowInactive(False)
    self.wave.setPen(None, amplify, attenuate)
    self.iniBtns()
    ev = self
    ev.sender = self
    ev.action = 0
    self.onActiveChanged(ev)
  def clrPlots(self, keepParent = False):
    # Note: closing the other way reveals a pyqtgraph bug for which docks with titles undock rather than close
    pywavescat.clrPlots(self, True) # no single scatter to clear
    pyscat3.clrPlots(self, True) # do no close Button Dock as we have multiple here to close
    if not(self.newparent) or keepParent or self.area is None: return
    self.area.remove()
    del self.area
    self.area = None
  def iniBtns(self):
    self.Dock = [None] * 3
    self.Bbox = [None] * 3
    for i in range(3):
      self.Dock[i] = dock()
      self.Bbox[i] = self.Dock[i].addBbox()
      if i < 2:
        self.area.add(self.Dock[i], 'bottom', self.docks[i])
      else:
        self.area.add(self.Dock[i], 'bottom')
      if i == 0:
        for j in range(8):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnwidth)
          hicol = np.array(self.logCols[j], dtype = float)*255.
          locol = hicol * self.colgr
          self.Bbox[i].setCols(j, hicol, locol)
        self.Bbox[i].Connect(0, self.pickCol0)
        self.Bbox[i].Connect(1, self.pickCol1)
        self.Bbox[i].Connect(2, self.pickCol2)
        self.Bbox[i].Connect(3, self.pickCol3)
        self.Bbox[i].Connect(4, self.pickCol4)
        self.Bbox[i].Connect(5, self.pickCol5)
        self.Bbox[i].Connect(6, self.pickCol6)
        self.Bbox[i].Connect(7, self.pickCol7)
      if i == 1:
        for j in range(8):
          self.Bbox[i].addButton()
          self.Bbox[i].setWidth(j, self.btnwidth)
        self.Bbox[i].Connect(0, self.toggleAccept0)
        self.Bbox[i].Connect(1, self.toggleAccept1)
        self.Bbox[i].Connect(2, self.toggleAccept2)
        self.Bbox[i].Connect(3, self.toggleAccept3)
        self.Bbox[i].Connect(4, self.toggleAccept4)
        self.Bbox[i].Connect(5, self.toggleAccept5)
        self.Bbox[i].Connect(6, self.toggleAccept6)
        self.Bbox[i].Connect(7, self.toggleAccept7)
      elif i == 2:
        self.Bbox[i].addButton(QtCancel)
        self.Bbox[i].addButton(QtApply)
        self.Bbox[i].setWidth(0, self.btnWidth)
        self.Bbox[i].setWidth(1, self.btnWidth)
        self.Bbox[i].Connect(0, self.cancelClick)
        self.Bbox[i].Connect(1, self.applyClick)
  def discriminate(self):
    pyscat3.discriminate(self)
    pywavescat.discriminate(self)
  def setBtns(self, spec = 0):
    pyscat3.setBtns(self, spec, 1)
    if spec == 2: return
    self.nlogic = np.zeros(8, dtype = int)
    for i in range(8):
      self.nlogic[i] = len(np.nonzero(np.logical_and(self.logic8[i], self.wave.active[0]))[0])
      self.Bbox[0].setText(i, self.nlogic[i])
  def setWaves(self, spec = 0): # we also handle sizes here. #spec = 0:both, 1:scats only, 2:picks only
    if spec == 2: # check we can get away with only showing picks
      if not(np.all(self.wave.active[2] == self.wact1)):
        spec = 0 # no we can't
    waveactive = np.logical_or(self.on, self.wave.active[2])
    self.wave.setactive([None, waveactive, None])
    self.wave.setPens(self.pens)
    self.pick.setPens(self.pens)
    if spec == 2:
      self.wave.setPick()
    else:
      self.wave.setWave()
      self.wact1 = np.copy(self.wave.active[1])
    self.pick.setWave() 
  def toggleAccept(self, i):
    pyscat3.toggleAccept(self, i)
    ev = self
    ev.sender = self.wave
    ev.action = 1
    self.onActiveChanged(ev)
    self.wave.plot.gbox.setFocus() # transfer focus to enable key press detection
  def pickCol0(self, ev = None):
    return self.pickCol(0)
  def pickCol1(self, ev = None):
    return self.pickCol(1)
  def pickCol2(self, ev = None):
    return self.pickCol(2)
  def pickCol3(self, ev = None):
    return self.pickCol(3)
  def pickCol4(self, ev = None):
    return self.pickCol(4)
  def pickCol5(self, ev = None):
    return self.pickCol(5)
  def pickCol6(self, ev = None):
    return self.pickCol(6)
  def pickCol7(self, ev = None):
    return self.pickCol(7)
  def pickCol(self, i = None):
    if i is None: return
    self.pick.setactive([None, None, np.logical_and(self.logic8[i], self.wave.active[0])])
    ev = self
    ev.sender = self.pick
    ev.action = 2
    self.onActiveChanged(ev)
    self.wave.plot.gbox.setFocus() # transfer focus to enable key press detection
  def onActiveChanged(self, ev):   # we write this carefully to optimise performance
    if self.on is None: self.on = np.ones(self.n, dtype = bool) # initialise self.on
    k = None
    if ev.sender == self.scats[0]: k = 0
    if ev.sender == self.scats[1]: k = 1
    if ev.sender == self.scats[2]: k = 2
    I = range(3)
    if ev.action != 2: # discriminate if necessary
      self.discriminate()
      if k is not None: # we need to set the scatters to work out the new colours
        self.setScats(ev.action)
    if k is None:                            # i.e. event occurred outside scatters
      if ev.sender == self.pick:             # inside pick
        if ev.action == 2:
          self.pick.setActive([None, self.pick.Active[2], False]) 
          self.wave.setActive([None, None, self.pick.Active[1]]) 
          for self.scat in self.scats:
            self.scat.setActive([None, None, self.pick.active[1]])
        elif ev.action == 0: # a master chooses, a slave obeys    
          self.wave.keyPressEvent(ev)
        else:
          print("Warning: Unknown call from self.pick") # should have no discrimination
      elif ev.sender == self.wave:           # inside wave
        if ev.action == 2:                   # just picking
          self.pick.setActive([None, self.wave.active[2], None])
          for self.scat in self.scats:
            self.scat.setActive([None, None, self.wave.active[2]])
        elif ev.action == 0: # self.wave is the master; the rest are slaves
          self.active = self.wave.active
          self.pick.setActive([self.wave.active[0], None, None]) 
          for self.scat in self.scats:
            self.scat.setActive([self.wave.active[0], None, None])
        else:  
          self.active = self.wave.active     # just discriminating
      elif ev.sender == self:    # first plot
        self.setScats(ev.action) # this is needed to define colours
        # Need to update waveplots to reflect scatter changes
      else:
        print("Unknown sender ID: " + str(ev.sender))
    else:                    # insider scatter
      if ev.action == 2:  # just picking
        I = []
        for i in range(3):
          if i != k:
            self.scats[i].active[2] = np.copy(self.scats[k].active[2])
            I.append(i)
        self.scat = self.scats[k]
        self.pick.setactive([None, self.scat.active[2], None])
        self.wave.setactive([None, None, self.scat.active[2]])
      elif ev.action == 0:   # delete/insert
        self.wave.keyPressEvent(ev)
      else:  
        pass # just discriminating
    self.setWaves(ev.action) 
    if ev.action == 2 or k is None: 
      self.setScats(ev.action, I)
    self.setBtns(ev.action)
    return k
  def applyClick(self):
    self.clrPlots()
    self.active = np.vstack((self.wave.active[0], self.on, self.wave.active[2])) # guarantee up-to-date
    return pyscat3.applyClick(self)
  def cancelClick(self):
    self.clrPlots()
    self.active = self.activeorig 
    return pyscat3.cancelClick(self)

PYDISCSPEC =  [['DDD', 0, -1, 'Index'],
               ['DDD', 1, -1, 'Positive deflection'],
               ['DDD', 2, -1, 'Negative deflection'],
               ['RFA', 0, -1, 'Peak/trough value'],
               ['RFA', 1, -1, 'Peak/trough time'],
               ['RFA', 6, -1, '20-80 rise time'],
               ['RFA', 7, -1, '80-20 fall time'],
               ['RFA', 17, -1, 'Decay fit offset constant'],
               ['RFA', 19, -1, 'Decay fit amplitude coefficient'],
               ['RFA', 21, -1, 'Decay fit log decay constant'],
               ['PCA', 0, -1, 'PCA1'], 
               ['PCA', 1, -1, 'PCA2'], 
               ['PCA', 2, -1, 'PCA3'], 
               ['PCA', 3, -1, 'PCA4'],
               ['DFT', 2, -1, 'FFT1 (Real)'], 
               ['DFT', 3, -1, 'FFT1 (Imag)'], 
               ['DFT', 4, -1, 'FFT2 (Real)'], 
               ['DFT', 5, -1, 'FFT2 (Imag)']] 

PYDISCDEFS = [[1, 2], [10, 11], [14, 15]]

class pydisc1(pywavescat):
  defanal = ['DDD', 'DDD']
  defspec = [1, 2]
  defcens = [-1, -1]
  btnWidth = 100
  def __init__(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], _anal = None, _spec = None,
      _cens = None, **kwds):
    pywavescat.__init__(self, _data, _chinfo, _lat, _Active, [], []) # the absent X, Y arguments are harmless
    self.setAnal(_anal, _spec, _cens, **kwds)
    self.setDoneFuncs()
  def setDoneFuncs(self, _doneFuncs = None):
    if not(isarray(_doneFuncs)): _doneFuncs = [_doneFuncs]
    if len(_doneFuncs) == 1: _doneFuncs = [_doneFuncs[0], None]
    self.doneFuncs = _doneFuncs
  def applyClick(self):
    self.clrPlots()
    if self.doneFuncs[0] is None:
      self.clrPlots()
      print("Apply Button Clicked.")
      return
    self.doneFuncs[0]()
  def cancelClick(self):
    self.active = self.activeorig 
    self.clrPlots()
    if self.doneFuncs[0] is None:
      self.clrPlots()
      print("Cancel Button Clicked.")
      return
    self.doneFuncs[1]()
  def iniBtns(self):
    if not(self.useDocks): return
    self.Dock = dock()
    self.area.add(self.Dock, 'bottom')
    self.Bbox = self.Dock.addBbox()
    self.Bbox.addButton(QtCancel)
    self.Bbox.addButton(QtApply)
    self.Bbox.setWidth(0, self.btnWidth)
    self.Bbox.setWidth(1, self.btnWidth)
    self.Bbox.Connect(0, self.cancelClick)
    self.Bbox.Connect(1, self.applyClick)
  def setApply(self, applytxt = None):
    if applytxt is None: applytxt = str(len(np.nonzero(self.on)[0]))
    self.Bbox.setText(1, applytxt)  
  def setPlots(self, I = range(3), *args, **kwds):
    pywavescat.setPlots(self, I, *args, **kwds)
    self.wave.setShowInactive(False)
    self.activeorig = np.copy(self.wave.active) 
    self.iniBtns() 
    self.setScat()
    self.discriminate()
    self.setApply() 
    ev = self
    ev.sender = self
    ev.action = 0
    self.onActiveChanged(ev)
  def clrPlots(self, keepParent = False):
    pywavescat.clrPlots(self, True)
    if not(self.useDocks): return
    self.area.clrDocks(self.Dock)
    self.Dock = None
    self.Bbox = None
    if not(self.newparent) or keepParent or self.area is None: return
    self.area.remove()
    del self.area
    self.area = None
  def setAnal(self, _anal = None, _spec = None, _cens = None, **kwds): 
    if _anal is None: _anal = self.defanal 
    if _spec is None: _spec = self.defspec 
    if _cens is None: _cens = self.defcens 
    self.anal, self.spec, self.cens = _anal, _spec, _cens
    if not(len(self.data)): return
    si = 1
    if 'si' in kwds: si  = kwds['si']
    self.Anal = [None] * 2
    self.XY = [[]] * 2
    self.xy = [""] * 2
    acs = [[]] * 2 # analspeccens record to obviate redundant calculations
    pqbar = pgb("Calculating scatter values ...", 6)
    for i in range(2):
      pqbar.set(i)
      self.Anal[i] = fpanal(self.data, self.anal[i], self.spec[i], self.cens[i])
      calc = i == 0
      _anal = self.anal[i]
      if type(_anal) is str:
        _anal = self.Anal[i].analkeys[_anal]
      acs[i] = np.hstack((_anal, self.cens[i]))
      k = -1
      if not(calc):
        for j in range(i):
          if k < 0:
            if len(acs[i]) == len(acs[j]):
              if np.all(acs[i] == acs[j]):
                k = j
        calc = k < 0
      if calc:                
        self.Anal[i].analyse(si)
      else:
        self.Anal[i].Z = self.Anal[k].Z
        self.Anal[i].Lbl = self.Anal[k].Lbl
        self.Anal[i].z = np.ravel(np.array(self.Anal[i].Z[:, self.spec[i]], dtype = float))
        self.Anal[i].lbl = self.Anal[i].Lbl[self.spec[i]]
      self.XY[i] = self.Anal[i].retRes()
      self.xy[i] = self.Anal[i].retLbl()
    pqbar.reset()
    pywavescat.iniScat(self, self.XY[0], self.XY[1], self.xy)
  def onActiveChanged(self, ev):
    pywavescat.onActiveChanged(self, ev)
    self.setApply()

class pydisc3(pywavescat3):
  defanal = ['DDD', 'DDD', 'PCA', 'PCA', 'DFT', 'DFT']
  defspec = [1, 2, 0, 1, 2, 3]
  defcens = [-1, -1, -1, -1, -1, -1]
  def __init__(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], _anal = None, _spec = None,
      _cens = None, **kwds):
    pywavescat3.__init__(self, _data, _chinfo, _lat, _Active, [], []) # the absent XXX, YYY arguments are harmless
    self.setAnal(_anal, _spec, _cens, **kwds)
  def setAnal(self, _anal = None, _spec = None, _cens = None, **kwds): 
    if _anal is None: _anal = self.defanal 
    if _spec is None: _spec = self.defspec 
    if _cens is None: _cens = self.defcens 
    self.anal, self.spec, self.cens = _anal, _spec, _cens
    if not(len(self.data)): return
    si = 1
    if 'si' in kwds: si  = kwds['si']
    self.Anal = [None] * 6
    self.XY = [[]] * 6
    self.xy = [""] * 6
    acs = [[]] * 6 # analspeccens record to obviate redundant calculations
    pqbar = pgb("Calculating scatter values ...", 6)
    for i in range(6):
      pqbar.set(i)
      self.Anal[i] = fpanal(self.data, self.anal[i], self.spec[i], self.cens[i])
      calc = i == 0
      _anal = self.anal[i]
      if type(_anal) is str:
        _anal = self.Anal[i].analkeys[_anal]
      acs[i] = np.hstack((_anal, self.cens[i]))
      k = -1
      if not(calc):
        for j in range(i):
          if k < 0:
            if len(acs[i]) == len(acs[j]):
              if np.all(acs[i] == acs[j]):
                k = j
        calc = k < 0
      if calc:                
        self.Anal[i].analyse(si)
      else:
        self.Anal[i].Z = self.Anal[k].Z
        self.Anal[i].Lbl = self.Anal[k].Lbl
        self.Anal[i].z = np.ravel(np.array(self.Anal[i].Z[:, self.spec[i]], dtype = float))
        self.Anal[i].lbl = self.Anal[i].Lbl[self.spec[i]]
      self.XY[i] = self.Anal[i].retRes()
      self.xy[i] = self.Anal[i].retLbl()
    pqbar.reset()
    pyscat3.setData(self, [self.XY[0], self.XY[2], self.XY[4]],
                          [self.XY[1], self.XY[3], self.XY[5]], 
                          self.Active,
                          [[self.xy[0], self.xy[1]],
                           [self.xy[2], self.xy[3]],
                           [self.xy[4], self.xy[5]]])

