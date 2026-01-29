import numpy as np
import pyqtgraph as pg
import pyclamp.gui.pyqtplot as pq
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.fpanal import *
from pyclamp.gui.xyfunc import *
from pyclamp.gui.pyplot import *
from pyclamp.gui.pydisc import *

class pywavescattabl (pywavescat) : # combines a pywavescat (wave + prev + scat) with a table
  row = None
  col = None
  val = None
  xlb = None
  ylb = None
  def __init__(self, _data = [], _chinfo = [], _lat = None, _Active = [True, True, False], 
      _Val = None, _Xlb = None, _Ylb = None):
    self.setData(_data, _chinfo, _lat, _Active, _Val, _Xlb, _Ylb)
  def setData(self, _data = [], _chinfo = [], _lat = [], _Active = [True, True, False],
      _Val = None, _Xlb = None, _Ylb = None):
    self.iniPick(_data, _chinfo, _lat, [_Active[0], False, False])
    self.iniWave(_data, _chinfo, _lat, _Active)
    self.iniTabl(_Val, _Xlb, _Ylb)
  def iniTabl(self, _Val = None, _Xlb = None, _Ylb = None):
    if isinstance(_Val, listtable):
      self.Val = _Val.readData()
      self.Xlb = _Val.readFlds()
      self.Ylb = _Val.readKeys()
    else:
      self.Val = _Val
      self.Xlb = _Xlb
      self.Ylb = _Ylb
    self.nrow = 0
    self.ncol = 0
    if self.Val is not None:
      self.nrow = len(self.Val)
      if self.nrow:
        self.ncol = len(self.Val[0])
      if self.Xlb is None: self.Xlb = range(self.ncol)
      if len(self.Xlb) != self.ncol:
        raise ValueError("Data and x label inputs incommensurate.")
      if self.Ylb is None: self.Ylb = range(self.nrow)
      if len(self.Ylb) != self.nrow:
        raise ValueError("Data and y label inputs incommensurate.")
    self.ind = np.arange(self.nrow, dtype = int)
    self.setRow()
    self.setCol()
    if not(self.nrow) or not(self.ncol): return
    self.tabl = pq.tabl() 
    self.tabl.setSortingEnabled(False)
    self.tabl.setData(self.Val)
    lbl = [[]] * self.ncol
    for j in range(self.ncol):
      lbl[j] = str(self.Xlb[j])
    self.tabl.setHorizontalHeaderLabels(lbl)
    lbl = [[]] * self.nrow
    for i in range(self.nrow):
      lbl[i] = str(self.Ylb[i])
    self.tabl.setVerticalHeaderLabels(lbl)
    self.tabl.setMouseClickEventFunc(self.onTablClick)
  def iniScat(self, _X = [], _Y = [], _xyl = None):
    if len(_X):
      if self.active is None:
        raise ValueError("Wave data must be initialised before scatter data.")
    self.scat = pyscat(_X, _Y, self.active)
    self.scat.setLabels(_xyl)
  def setRow(self, _row = None):
    self.row = _row
    if self.row is None: return
    self.ylb = self.Ylb[self.row]
    z = np.zeros(self.nrow, dtype = bool)
    z[self.row] = True
    return z
  def setCol(self, _col = 0):
    self.col = _col
    if self.Val is None: return
    self.xlb = self.Xlb[self.col]
    self.val = [None] * self.nrow
    for i in range(self.nrow):
      if len(self.Val[i]) > self.col:
        self.val[i] = self.Val[i][self.col]
    if self.scat is None: 
      self.iniScat(self.ind, self.val)
    else:
      self.scat.setData(self.ind, self.val, self.scat.active)
      self.scat.setLabels(["#", self.xlb])
  def setPlots(self, I = range(4), *args, **kwds):
    '''
    if self.area is None: self.setArea(*args, **kwds)
    if not(self.useDocks): 
      if listfind(I, 0) is not None:
        self.pick.setPlot(parent = self.gbox, row = 0)
        self.pick.setActiveChangedFunc(self.onActiveChanged)
        self.pick.toggleOverlay()
      if listfind(I, 1) is not None:  
        self.wave.setPlot(parent = self.gbox, row = 0)
        self.wave.setActiveChangedFunc(self.onActiveChanged)
        self.wave.toggleOverlay()
      if listfind(I, 2) is not None:
        self.gbox.nextRow()
        self.scat.setPlot(parent = self.gbox, row = 1)
        self.scat.setActiveChangedFunc(self.onActiveChanged)
      if listfind(I, 3) is not None:  
        self.tabl.setParent(self.gbox)
    else:
      self.docks = [None] * 4
      self.boxes = [None] * 4 
      dockTitles = ['', '', '', '']
      pickwavescat = [self.pick, self.wave, self.scat]
      for i in I:
        self.docks[i] = dock(dockTitles[i])
        self.boxes[i] = gbox()
        if i < 3:
          self.docks[i].add(self.boxes[i])
        else:
          self.docks[i].add(self.tabl)
        if i % 2:
          self.area.add(self.docks[i], 'right', self.docks[i-1])
        else:
          self.area.add(self.docks[i])
        if i < 3:
          pickwavescat[i].setPlot(parent = self.boxes[i])
          pickwavescat[i].setActiveChangedFunc(self.onActiveChanged)
          if i < 2: 
            pickwavescat[i].toggleOverlay()
            pickwavescat[i].keysDisabled = True
      self.setCol(self.col) # auto-update axis
    '''
    _I = np.array(I)
    pywavescat.setPlots(self, list(_I[_I < 3]), *args, **kwds)
    if listfind(I, 3) is None: return self.parent
    if not(self.useDocks):
      self.tabl.setParent(self.gbox)
    elif listfind(I, 3):
      self.docks.append(None)
      self.boxes.append(None)
      self.docks[-1] = dock('')
      self.docks[-1].add(self.tabl)
      self.area.add(self.docks[-1], 'right', self.docks[-2])
    self.setCol(self.col) # auto-update axis -> this line hid rubber bands for some reason
    return self.parent
  def setScat(self, _ = False):
    if self.on is None:
      self.scat.setEllipse(_)
    self.scat.setScat()
    self.on = np.logical_and(self.wave.active[0], self.scat.active[1])
  def discriminate(self):
    self.on = np.logical_and(self.wave.active[0], self.scat.active[1])
  def onTablClick(self, ev):
    if not(isint(ev.row)) or not(isint(ev.col)): return
    #print(self.tabl.selectedItems())
    if self.row is None: # for the first time we need to repeat
      active2 = self.setRow(ev.row)
      self.scat.setActive([None, None, active2])
      self.setCol(ev.col) # this changes the scatter completely
      ev.sender = self.scat
      ev.action = 2
      pywavescat.onActiveChanged(self, ev)
    active2 = self.setRow(ev.row)
    self.scat.setActive([None, None, active2])
    self.setCol(ev.col) # this changes the scatter completely
    ev.sender = self.scat
    ev.action = 2
    pywavescat.onActiveChanged(self, ev)
  def onActiveChanged(self, ev):
    pywavescat.onActiveChanged(self, ev)
    active2 = self.scat.active[2]
    i = np.nonzero(active2)[0]
    if len(i) == 1:
      self.setRow(i[0])
      if isint(self.col):
        self.tabl.setCurrentCell(self.row, self.col)

class pysumm (pywavescattabl):
  doneFuncs = [None, None]
  btnWidth = 100
  Dock = None
  Bbox = None
  def setDoneFuncs(self, _doneFuncs = None):
    if not(isarray(_doneFuncs)): _doneFuncs = [_doneFuncs]
    if len(_doneFuncs) == 1: _doneFuncs = [_doneFuncs[0], None]
    self.doneFuncs = _doneFuncs
  def setPlots(self, I = range(4), *args, **kwds):
    _ = pywavescattabl.setPlots(self, I, *args, **kwds)
    self.iniBtns()
    return _
  def clrPlots(self, keepParent = False):
    pywavescattabl.clrPlots(self)
    self.area.clrDocks(self.Dock)
    self.Dock = None
    self.Bbox = None
    if not(self.newparent) or keepParent or self.area is None: return
    self.area.remove()
    del self.area
    self.area = None
  def iniBtns(self):
    if not(self.useDocks): return
    self.Dock = dock()
    self.Bbox = self.Dock.addBbox()
    self.area.add(self.Dock, 'bottom')
    i = 0
    self.Bbox.addButton(QtApply)
    self.Bbox.setText(i, 'OK')
    self.Bbox.setWidth(i, self.btnWidth)
    self.Bbox.Connect(i, self.OK)
    i += 1
    self.Bbox.addButton(QtCancel)
    self.Bbox.setText(i, 'Cancel')
    self.Bbox.setWidth(i, self.btnWidth)
    self.Bbox.Connect(i, self.CC)
  def OK(self, ev = None):
    self.clrPlots()
    if self.doneFuncs[0] is None:
      print("User Clicked OK")
      return None
    else:
      return self.doneFuncs[0](ev)
  def CC(self, ev = None):
    self.clrPlots()
    if self.doneFuncs[1] is None:
      print("User Cancelled")
      return None
    else:
      return self.doneFuncs[1](ev)

