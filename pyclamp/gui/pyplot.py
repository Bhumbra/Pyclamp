import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyclamp.gui.pyqtplot import *
import pyclamp.gui.lbwgui as lbw
from pyclamp.gui.pgb import pgb
from pyclamp.gui.xyfunc import *
from pyclamp.dsp.channel import chWave
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.wffunc import minmax2
from pyclamp.dsp.fpfunc import *

def SetJetRGBC(_JetRGBC = None): # default None (0.5 works well with black foreground on white backgroun)
  global JetRGBC
  JetRGBC = _JetRGBC
  return JetRGBC

def SetDefScatMarkerSize(_DEFSCATMARKERSIZE = None): # pyscat defaults to 3.
  global DEFSCATMARKERSIZE
  DEFSCATMARKERSIZE = _DEFSCATMARKERSIZE
  return DEFSCATMARKERSIZE

JetRGBC = SetJetRGBC()
DEFSCATMARKERSIZE = SetDefScatMarkerSize()

BaseFormClass = QtWidgets.QMainWindow
BaseGboxClass = gbox
BasePlotClass = pg.PlotItem
QtGuiQPen = QtGui.QPen
QtGuiQBrush = QtGui.QBrush

QtNoButton = 0
QtLeftButton = 1
QtRightButton = 2
QtMidButton   = 4
QtKeySpace   = 32 # ASC #084 = lower-case T
QtKeyEscape   = 16777216 # ESCAPE
QtKeyInsert   = 16777222 # INSERT
QtKeyDelete   = 16777223 # DELETE
QtKeyShift    = 16777248 # CONTROL
QtKeyControl  = 16777249 # SHIFT
QtKeyMeta     = 16777250 # META
QtKeyAlternate= 16777251 # ALTERNATE
QtMapModifiers= { QtControlModifier:QtKeyControl,
                  QtShiftModifier:QtKeyShift,
                  QtMetaModifier:QtKeyMeta,
                  QtAlternateModifier:QtKeyAlternate }
KEY_ALL_INVERT_NONE = [85, 73, 79] # U I O

QtDotLine = QtCore.Qt.PenStyle.DotLine

# ABXY: Append select, Block select, X, Y

ABXYKeyModifiers = QtKeyModifiers[:]
ABXYKeyMod = [QtMapModifiers[ABXYKeyModifiers[0]], QtMapModifiers[ABXYKeyModifiers[1]],\
              QtMapModifiers[ABXYKeyModifiers[2]], QtMapModifiers[ABXYKeyModifiers[3]]]
ABKeyMod, XYKeyMod = [ABXYKeyMod[0], ABXYKeyMod[1]], [ABXYKeyMod[2], ABXYKeyMod[3]]

def nullpen(x = None):
  return pg.mkPen(x)

def nullbrush(x = None):
  return pg.mkBrush(x)

def jetrgb(x):
  r = 0.0
  g = 0.0
  b = 0.0
  if x < 0:
      b = 0.5
  elif x < 0.125:
      b = 0.5 + 4.0 * x
  elif x < 0.375:
      g = 4.0 * (x - 0.125)
      b = 1.0
  elif x < 0.625:
      r = 4.0 * (x - 0.375)
      g = 1.0
      b = 4.0 * (0.625 - x)
  elif x < 0.875:
      r = 1.0
      g = 4.0 * (0.875 - x)
  elif x < 1:
      r = 0.5 + 4.0 * (1.0 - x)
  else:
      r = 0.5
  if JetRGBC is None:
    return np.array((r,g,b), dtype = float)
  return np.minimum(JetRGBC*np.array((r,g,b), dtype = float), 1.)


def jetpen(_x, _w = None):
  if isfloat(_x):
    rgb = 255.*jetrgb(_x)
    p = pg.mkPen(color=(rgb[0], rgb[1], rgb[2]))
    if _w is not None: p.setWidth(_w)
    return p
  elif isint(_x):
    p = [None] * _x
    x = np.linspace(1., 0., _x)
    for i in range(_x):
      p[i] = jetpen(x[i], _w)
    return p
  elif type(_x) is PyQt4.QtGui.QPen:
    if _w is None: return _x
    pn = pg.pkPen(_x)
    pn.setWidth(_w)
    return pn
  elif elType(_x) is PyQt4.QtGui.QPen:
    if _w is None: return _x
    n = len(_x)
    p = np.tile(nullpen(), n)
    for i in range(n):
      p[i] = pg.mkPen(_x[i])
      p[i].setWidth(_w)
    return p
  else:
    sX = _x.shape
    x = np.ravel(_x)
    n = len(x)
    p = np.tile(nullpen(), n)
    for i in range(n):
      p[i] = jetpen(x[i])
      p[i].setWidth(_w)
    return p.reshape(sX)

def jetbrush(_x):
  if isfloat(_x):
    rgb = 255.*jetrgb(_x)
    return pg.mkBrush(color=(rgb[0], rgb[1], rgb[2]))
  else:
    p = [None] * _x
    x = np.linspace(1., 0., _x)
    for i in range(_x):
      p[i] = jetbrush(x[i])
    return p

def rgbPen(x, c = None):
  if c is None: c = 255.
  if type(x) is str: return pg.mkPen(x)
  return pg.mkPen(color=(min(255., c*x[0]), min(255., c*x[1]), min(255., c*x[2])))

def rgbBrush(x, c = None):
  if c is None: c = 255.
  if type(x) is str: return pg.mkBrush(x)
  return pg.mkBrush(color=(min(255., c*x[0]), min(255., c*x[1]), min(255., c*x[2])))

def amplify(x, d = 0.333, c = 2): # amplifies a pen/brush colour
  rgb = x.color().getRgb()
  w = x.width() * c
  rgb = np.array(rgb, dtype = float)[:3] / 255.
  sumc = rgb.sum()
  if sumc > 1.999:
    rgb -= d
  else:
    rgb += d
  for i in range(3):
    rgb[i] = max(0., min(1., rgb[i]))
  if type(x) is QtGuiQPen:
    return pg.mkPen(color=(rgb[0]*255., rgb[1]*255., rgb[2]*255.), width = w)
  if type(x) is QtGuiQBrush:
    return pg.mkBrush(color=(rgb[0]*255., rgb[1]*255., rgb[2]*255.), width = w)

def attenuate(x, d = 0.333, c = 0.5, a = 0.333, penstyle = QtDotLine): # attenuates a pen/brush colour
  rgb = x.color().getRgb()
  rgb = np.array(rgb, dtype = float)[:3] / 255.
  sumc = rgb.sum()
  if sumc > 1.999:
    rgb -= d
  else:
    rgb += d
  for i in range(3):
    rgb[i] = max(0., min(1., rgb[i]))
  if type(x) is QtGuiQPen:
    return pg.mkPen(color=(c*rgb[0]*255., c*rgb[1]*255., c*rgb[2]*255., a*255.), style = penstyle)
  if type(x) is QtGuiQBrush:
    return pg.mkBrush(color=(c*rgb[0]*255., c*rgb[1]*255., c*rgb[2]*255.))

def enlarge(x, c = 1.5):
  return x * c

def shrink(x, c = 0.6666666666666666667):
  return x * c

class pywave:
  defpads = [0., 0.10]
  defpen = 'jet'
  defoverlay = False
  jetpens = None
  pens = None
  Active = None
  xxyy = None
  plot = None
  plots = None
  picks = None
  mm2 = None
  pgbar = None
  overlay = None
  Marks = None
  labels = None
  lblpos = None
  deflblpos = ['bottom', 'left']
  showInactive = True
  keysDisabled = False
  def __init__(self, _data = [], _chinfo = [], _onsets = [], _Active = [True, True, False]):
    self.setData(_data, _chinfo, _onsets, _Active)
    self.setViewRangeChangedFunc()
    self.setMouseClickEventFunc()
    self.setKeyPressEventFunc()
    self.setActiveChangedFunc()
  def setViewRangeChangedFunc(self, _viewRangeChangedFunc = None):
    self.viewRangeChangedFunc = _viewRangeChangedFunc
  def setMouseClickEventFunc(self, _mouseClickEventFunc = None):
    self.mouseClickEventFunc = _mouseClickEventFunc
  def setKeyPressEventFunc(self, _keyPressEventFunc = None):
    self.keyPressEventFunc = _keyPressEventFunc
    self.keyUIOmap = {KEY_ALL_INVERT_NONE[0]:self.selectall,
                      KEY_ALL_INVERT_NONE[1]:self.selectinvert,
                      KEY_ALL_INVERT_NONE[2]:self.selectnone}
  def setActiveChangedFunc(self, _activeChangedFunc = None):
    self.activeChangedFunc = _activeChangedFunc
  def setData(self, _data = [], _chinfo = [], _onsets = [], _Active = [True, True, False]):
    self.data = _data
    self.chinfo = _chinfo
    self.onsets = _onsets
    self.N = len(self.data)
    self.n = np.zeros(self.N, dtype = int)
    if not(self.N): return
    self.setChInfo(self.chinfo)
    if nDim(self.data) == 1: self.data = [self.data]
    if nDim(self.data) == 2: self.data = [self.data]
    self.N = len(self.data)
    self.n = np.zeros(self.N, dtype = int)
    self.ns = len(self.data[0][0])
    self.ne = 0
    for i in range(self.N):
      self.n[i] = len(self.data[i])
      self.ne += self.n[i]
    self.setOnsets(_onsets)
    self.mint = self.Onsets[0] * self.si
    self.maxt = self.ns * self.si
    self.endt = (self.Onsets[-1] + self.ns) * self.si
    self.setMarks()
    self.setVisual(_Active)
  def setChInfo(self, _chinfo = None):
    self.chinfo = chWave()
    self.chinfo.index = 0
    self.chinfo.name = ''
    self.chinfo.units = ''
    self.chinfo.samplint = None
    self.chinfo.gain = 1
    self.chinfo.offset = 0
    if type(_chinfo) is int or type(_chinfo) is float:
      self.chinfo.samplint = _chinfo
      self.chinfo.gain = 1
      self.chinfo.offset = 0
    elif type(_chinfo) is list:
      if len(_chinfo) > 0:
        if type(_chinfo[0]) is int or type(_chinfo[0]) is float:
          self.chinfo.samplint = _chinfo[0]
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
      if len(_chinfo) > 1:
        if type(_chinfo[1]) is int or type(_chinfo[1]) is float:
          self.chinfo.gain = _chinfo[1]
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
      if len(_chinfo) > 2:
        if type(_chinfo[0]) is int or type(_chinfo[0]) is float:
          self.chinfo.offset = _chinfo[2]
        else:
          raise ValueError("pywave: channel info lists must contain numeric data only.")
    elif isinstance(_chinfo, chWave):
      self.chinfo.index = _chinfo.index
      self.chinfo.name = _chinfo.name
      self.chinfo.units = _chinfo.units
      self.chinfo.samplint = _chinfo.samplint
      self.chinfo.gain = _chinfo.gain
      self.chinfo.offset = _chinfo.offset
    self.name = self.chinfo.name
    self.units = self.chinfo.units
    self.samplint = self.chinfo.samplint
    self.gain = self.chinfo.gain
    self.offset = self.chinfo.offset
    self.si = self.samplint
    if self.name is None: self.name = 'Value'
    if not(len(self.name)): self.name = 'Value'
    if self.units is None: self.units = 'units'
    if not(len(self.units)): self.units = 'units'
  def setOnsets(self, _onsets = []):
    if _onsets is None: _onsets = []
    if type(_onsets) is np.ndarray: _onsets = [_onsets]
    self.onsets = _onsets
    if not(self.N): return
    if not(len(self.onsets)):
      self.onsets = [None] * self.N
      k = -1
      for i in range(self.N):
        self.onsets[i] = np.empty(self.n[i], dtype = int)
        for j in range(self.n[i]):
          k += 1
          self.onsets[i][j] = k * int(self.ns)
    if len(self.onsets) != self.N:
      raise ValueError("Onsets data incommensurate with wave data")
    for i in range(self.N):
      if len(self.onsets[i]) != self.n[i]:
        raise ValueError("Onsets data incommensurate with wave data")
    self.Onsets = np.hstack(self.onsets)
  def setMarks(self, _marks = [], _markc = 'T', _markr = 1.025): #markil = [indices, onsetsencies]
    if _marks is None: _marks = []
    self.markc = _markc
    self.markr = _markr
    ndmarks = nDim(_marks)
    if len(_marks) == 0:
      self.marks = []
    elif ndmarks == 1:
      self.marks = np.zeros(2, len(_marks), dtype = int)
      self.marks[1,:] = _marks
    elif ndmarks == 2:
      self.marks = _marks
    else:
      raise ValueError("pywave.setMarks(): Unknown index input array specification.")
    self.markn = len(self.marks)
    if self.markn:
      self.markn = len(self.marks[0])
    self.remMarks()
  def remMarks(self):
    if self.plot is None: return
    if self.Marks is not None:
      for i in range(len(self.Marks)):
        self.plot.remove(self.Marks[i])
    self.Marks = None
  def setVisual(self, _Active = [True, True, False]):
    self.mm2 = minmax2(type(self.data[0][0][0]))
    self.minData = np.inf
    self.maxData = -np.inf
    self.Data = []             # Single 2D array containing all data
    self.eid = []              # episode ID in the form of two-length tuple list
    self.Eid = [None] * self.N # episode ID in the form of 2D list
    for i in range(self.N):
      self.Eid[i] = np.arange(self.n[i], dtype = int)
      for j in range(self.n[i]):
        self.eid.append([i, j])
        if self.ns != len(self.data[i][j]):
          raise ValueError("Data input includes episodes of dissimilar sizes")
        self.Data.append(self.data[i][j])
        self.minData = min(self.minData, float(self.data[i][j].min())*self.gain+self.offset)
        self.maxData = max(self.maxData, float(self.data[i][j].max())*self.gain+self.offset)
    self.mm2.setData(self.Data, [self.si, self.gain, self.offset], self.Onsets)
    self.mm2.calcMinMax()
    self.setActive(_Active) # this is the first time setActive() is called
    self.delList = []
    self.delMark = []
    self.setPen(self.defpen)      # default to jetpens
    if self.plot is None: return
    # self.setWave(self.defoverlay) - No! This must always be initiated from the self.setplot() side
  def iniLims(self, _lims = None, _pads = None):
    if _pads is None: _pads = self.defpads
    if _lims is None:
      if self.overlay:
        _lims = [[0., self.maxt], [self.minData, self.maxData]]
      else:
        _lims = [[self.mint, self.endt], [self.minData, self.maxData]]
    self.pads = _pads
    self.pmax = np.max(self.pads)
    self.lims = [[_lims[0][0], _lims[0][1]], [_lims[1][0], _lims[1][1]]]
    self.mids = [0.5*(self.lims[0][1] + self.lims[0][0]), 0.5*(self.lims[1][1] + self.lims[1][0])]
    self.wids = [0.5*(self.lims[0][1] - self.lims[0][0]), 0.5*(self.lims[1][1] - self.lims[1][0])]
    for i in range(2):
      if self.pads[i] != 0.:
        self.wids[i] *= (1.+self.pads[i])
        self.lims[i] = [self.mids[i] - self.wids[i], self.mids[i] + self.wids[i]]
    self.xxyy =  [[self.lims[0][0], self.lims[0][1]], [self.lims[1][0], self.lims[1][1]]]
    if self.plot is None: return
    self.plot.iniLims(self.xxyy)
    self.plot.update()
  def setLims(self, _xxyy = None):
    if self.plot is None: return
    chkx = False
    if _xxyy is None:
      xxyy = self.xxyy
    elif len(_xxyy) == 1:
      xxyy = [_xxyy[0], self.xxyy[1]]
      chkx = True
    else:
      if _xxyy[0] is None: _xxyy[0] = self.xxyy[0]
      if _xxyy[1] is None: _xxyy[1] = self.xxyy[1]
      xxyy =  [[_xxyy[0][0], _xxyy[0][1]], [_xxyy[1][0], _xxyy[1][1]]]
      chkx = True
    if chkx:
      if xxyy[0] is not None:
        if len(xxyy[0]):
          if elType(xxyy[0]) is int:
            for i in range(len(xxyy[0])):
              xxyy[0][i] *= self.si
    self.plot.updateRanges(xxyy)
  def delete(self):
    dellist = []
    delmark = []
    indmark = np.array([], dtype = int)
    markind = None
    if self.marks is not None:
      if len(self.marks):
        markind = self.marks[0,:]
    chMarks = markind is not None
    k = -1
    for i in range(self.N):
      for j in range(self.n[i]):
        k += 1
        if self.Active[2][i][j]:
          self.Active[0][i][j] = False
          self.Active[2][i][j] = False
          dellist.append([i, j])
          if chMarks:
            _indmark = np.array(np.nonzero(markind == k)[0], dtype = int)
            if len(_indmark):
              indmark = np.hstack( (indmark, _indmark) )
    if len(indmark):
      indmark = np.array(indmark, dtype = int)
      delmark = np.copy(self.marks[:, indmark])
      self.marks = np.delete(self.marks, indmark, axis = 1)
      if not len(self.marks): self.marks = None
      self.setMarks(self.marks, self.markc, self.markr)
    self.delList.append(dellist)
    self.delMark.append(delmark)
    self.setActive()
  def insert(self):
    if not(len(self.delList)): return
    for k in range(len(self.delList[-1])):
      ij = self.delList[-1][k]
      i, j = ij[0], ij[1]
      self.Active[0][i][j] = True
      self.Active[2][i][j] = True
    self.delList = self.delList[:-1]
    delmark = self.delMark[-1]
    if len(delmark):
      if self.marks is None:
        self.marks = np.copy(delmark)
      else:
        self.marks = np.hstack( (self.marks, delmark) )
        i = np.argsort(self.marks[0,:]*self.ns + self.marks[1,:])
        self.marks = self.marks[:, i]
      self.setMarks(self.marks, self.markc, self.markr)
    self.delMark = self.delMark[:-1]
    self.setActive()
  def selectall(self):
    for i in range(self.N):
      for j in range(self.n[i]):
        if self.Active[0][i][j]:
          self.Active[2][i][j] = True
    self.setActive()
  def selectinvert(self):
    for i in range(self.N):
      for j in range(self.n[i]):
        if self.Active[0][i][j]:
          self.Active[2][i][j] = not(self.Active[2][i][j])
    self.setActive()
  def selectnone(self):
    for i in range(self.N):
      for j in range(self.n[i]):
        if self.Active[0][i][j]:
          self.Active[2][i][j] = False
    self.setActive()
  def setShowInactive(self, _showInactive = True):
    self.showInactive = _showInactive
    self.setActive()
  def onActiveChanged(self, ev = None):
    if self.activeChangedFunc is None: return
    if ev is not None: ev.sender = self
    self.activeChangedFunc(ev)
  def setActive(self, _Active = [None, None, None]):
    # First index: Boolean logic for whether showable (i.e. alive) or not (i.e. dead)
    # Second index: Boolean logic for selected events
    # Third index: Boolean logic for picked events
    if not(self.N): return
    if self.Active is None: self.Active = [[], [], []]
    for h in range(len(_Active)):
      _active = _Active[h]
      if _active is None:
        pass
      elif type(_active) is bool:
        self.Active[h] = [[]] * self.N
        for i in range(self.N):
          self.Active[h][i] = np.tile(_active, self.n[i])
      elif type(_active) is list:
        self.Active[h] = [[]] * self.N
        for i in range(self.N):
          self.Active[h][i] = np.copy(_active[i])
      elif type(_active) is np.ndarray:
        self.Active[h] = [np.copy(_active)]
      else:
        self.Active[h] = [_active]
    self.setvisual()
  def setactive(self, _active):
    k = -1
    for i in range(self.N):
      for j in range(self.n[i]):
        k += 1
        if _active[0] is not None: self.Active[0][i][j] = _active[0][k]
        if _active[1] is not None: self.Active[1][i][j] = _active[1][k]
        if _active[2] is not None: self.Active[2][i][j] = _active[2][k]
    self.setvisual()
  def setvisual(self):
    self.visual = np.tile(False, self.ne)
    self.active = np.tile(False, (3, self.ne))
    k = -1
    h = 0 if self.showInactive else 1
    for i in range(self.N):
      for j in range(self.n[i]):
        k += 1
        g = 0
        self.active[g][k] = self.Active[g][i][j]
        for g in range(1, 3):
          self.active[g][k] = np.logical_and(self.active[0][k], self.Active[g][i][j])
        self.visual[k] = self.active[h][k]
    self.unpick = np.logical_and(self.visual, np.logical_not(self.active[2]))
    self.visord = np.arange(self.ne, dtype = int)
    self.invisi = np.nonzero(np.logical_not(self.visual))[0]
    if self.showInactive:
      deadind = np.nonzero(np.logical_not(self.active[0]))[0]
      offind = np.nonzero(np.logical_and(self.active[0], np.logical_not(self.active[1])))[0]
      onind = np.nonzero(np.logical_and(self.active[0], self.active[1]))[0]
      self.visord = np.argsort(np.hstack((deadind, offind, onind)))
  def cpyActive(self, _I = None):
    I = _I
    if I is None:
      I = range(len(self.Active))
    elif type(_I) is int:
      I = [I]
    _Active = [[]] * len(self.Active)
    for i in I:
      _Active[i] = [[]] * len(self.Active[i])
      for j in range(len(self.Active[i])):
        _Active[i][j] = np.copy(self.Active[i][j])
    if type(_I) is int: return _Active[i]
    return _Active
  def setPen(self, _pen = None, _ampfunc = amplify, _attfunc = attenuate):
    if _pen is None: _pen = self.pen
    self.pen = _pen
    self.ampfunc = _ampfunc
    self.attfunc = _attfunc
    if self.mm2 is None: return
    self.jetpens = jetpen(self.ne)
    self.setPens()
  def setPens(self, _pens = None, _w = None):
    if type(_pens) is int and _w is None:
      _pens, _w = None, _pens
    if _pens is None:
      if self.pens is not None and _w is not None:
        n = len(self.pens)
        for i in range(n):
          self.pens[i].setWidth(_w)
        return
    else:
      self.pen = None
      if not(isarray((_pens))): _pens = [_pens] * self.ne
      if _w is None:
        self.pens = _pens[:]
        return
      n = len(_pens)
      self.pens = [nullpen()] * n
      for i in range(n):
        self.pens[i] = pg.mkPen(_pens[i])
        self.pens[i].setWidth(_w)
      return
    if self.pen == 'jet':
      self.pens = self.jetpens[:]
    else:
      self.pens = np.array([self.pen]*self.ne)
    if self.mm2 is None: return
    for i in range(len(self.invisi)):
      self.pens[self.invisi[i]] = nullpen()
  def setPlot(self, _pgbar = None, _pgind = 1, *args, **kwds):
    if self.plot is None:
      self.plot = graph(*args, **kwds)
      self.parent = self.plot.gbox
    else:
      return self.resetPlot(_pgbar, _pgbind)
    self.plot.setViewRangeChangedFunc(self.viewRangeChanged)
    self.plot.setMouseClickEventFunc(self.mouseClickEvent)
    self.plot.setKeyPressEventFunc(self.keyPressEvent)
    self.plot.addRbFunc(self.rbEvent)
    self.setWave(self.defoverlay, _pgbar, _pgind)
    if self.labels is not None: self.setLabels(self.labels, self.lblpos)
    return self.plot
  def resetPlot(self, _pgbar = None, _pgind = 1):
    if self.plots is not None:
      while len(self.plots):
        self.plots[0].clear()
        self.plots[0].update()
        self.plot.remove(self.plots[0])
        del self.plots[0]
      self.plots = None
    if self.picks is not None:
      while len(self.picks):
        if self.picks[0] is not None:
          self.picks[0].clear()
          self.picks[0].update()
          self.plot.remove(self.picks[0])
        del self.picks[0]
      self.picks = None
    self.plot.clear()
    self.plot.update()
    self.clrPens()
    self.setWave(self.defoverlay, _pgbar, _pgind)
    return self.plot
  def updateData(self, *args): # change data without changing plots
    self.setData(*args)
    self.iniLims()
  def clrGraph(self):
    self.clrPens()
    self.clrPlots()
    self.clrPicks()
    self.clrMarks()
  def clrPens(self):
    self.pens = None
  def clrPlots(self):
    if self.plots is None: return
    while len(self.plots):
      del self.plots[0]
    self.plots = None
  def clrPicks(self):
    if self.picks is None: return
    while len(self.picks):
      del self.picks[0]
    self.picks = None
  def clrMarks(self):
    if self.marks is None: return
    while len(self.marks):
      del self.marks[0]
    self.marks = None
  def setLabels(self, xy = None, pos = None, **kwds):
    if type(xy) is tuple: xy=list(xy)
    if type(xy) is not list: xy=[xy]
    if len(xy) == 1: xy = [xy[0], None]
    if self.lblpos is None:
      self.lblpos = self.deflblpos if pos is None else pos
    else:
      if pos is not None: self.lblpos = pos
    self.labels = xy
    if self.plot is None: return
    self.plot.setLabels(self.labels, self.lblpos, **kwds)
  def setWave(self, _overlay = None, _pgb = None, _pgind = 1): # two plots: Plots then Picks
    if _pgb is not None:  self.pgbar = _pgb
    if self.mm2 is None or self.plot is None: return
    if _overlay is not None:
      self.overlay = _overlay
      if self.overlay:
        _lims = [[0., self.maxt], [self.minData, self.maxData]]
      else:
        _lims = [[0., self.endt], [self.minData, self.maxData]]
      if self.plots is not None:
        if len(self.plots) != self.ne:
          self.clrPens()
          self.clrPlots()
        else:
          for i in range(self.ne):
            if self.plots[i] is not None:
              self.plots[i].setData(_lims[0], _lims[1])
              self.plots[i].setPen(nullpen())
              self.plots[i].setVisible(False)
              self.plots[i].update()
        self.iniLims()
      if self.xxyy is not None:
        self.mm2.calcMinMax(self.xxyy[0][0], self.xxyy[0][1], self.overlay) # must calculate if overlay changed
    self.pgind = _pgind
    if self.pens is None: self.setPens()
    if self.plots is None:
      self.iniLims() # to set limits to initialise x and y values
      self.plot.clear()
      self.plot.update()
      self.plots = [None]*self.ne
      if self.pgbar is None:
        self.pgbar = pgb("Preparing plots ...", self.ne * self.pgind)
        self.pgind = 0
      else:
        self.pgind *= self.ne
      for i in range(self.ne):
        self.pgbar.set(i+self.pgind)
        self.plots[i] = pg.PlotCurveItem(self.lims[0], self.lims[1], pen=self.pens[i])
        self.plot.add(self.plots[i])
      self.iniLims() ## initialise limits to over-ride range changes
      self.setLims()
    if _pgb is None: # automatically close if constructed here
      self.pgbar.reset()
    self.T, self.Y = self.mm2.retXY()
    for i in range(self.ne):
      h = self.visord[i] if self.showInactive else i # previously created random crashes (used h = i before)
      if self.plots[h] is not None:
        self.plots[h].setVisible(self.unpick[i])
        pn = self.pens[i]
        if not(self.active[1][i]) and self.attfunc is not None:
          if self.attfunc is not None: pn = self.attfunc(pn)
        if self.unpick[i]:
          self.plots[h].setData(self.T[i], self.Y[i])
          self.plots[h].setPen(pn)
        else:
          self.plots[h].setData([], [])
          self.plots[h].setPen(nullpen())
    self.setPick()
    self.setMark()
    return self.plots, self.picks
  def setPick(self):
    # Picked waves
    if self.picks is not None:
      if len(self.picks) != self.ne:
        self.clrPicks()
    if self.picks is None:
      self.picks = [None]*self.ne
    for i in range(self.ne):
      pn = self.pens[i]
      if self.active[2][i]:
        if self.ampfunc is not None: pn = self.ampfunc(pn)
        if self.picks[i] is None:
          self.picks[i] = pg.PlotCurveItem(self.T[i], self.Y[i], pen=pn)
          self.plot.addItem(self.picks[i])
          self.picks[i].setVisible(self.visual[i])
        else:
          self.picks[i].setData(self.T[i], self.Y[i])
          self.picks[i].setPen(pn)
          self.picks[i].setVisible(self.visual[i])
      elif self.picks[i] is not None:
        self.picks[i].setData([], [])
        self.picks[i].setPen(nullpen())
        self.picks[i].setVisible(False)
    return self.picks
  def setMark(self):
    if not(self.markn): return
    if self.Marks is not None:
      if len(self.Marks) != self.markn:
        self.clrMarks()
    if self.Marks is None:
      self.Marks = [[]] * self.markn
      for i in range(self.markn):
        self.Marks[i] = pg.TextItem(self.markc, anchor = (0.5, 0.5))
        self.plot.add(self.Marks[i])
    self.markx = self.mm2.ind2X(self.marks)
    self.marky = self.markr * (self.maxData-self.minData) + self.minData
    for i in range(self.markn):
      self.Marks[i].setPos(self.markx[i], self.marky)
  def setOverlay(self, _overlay = None):
    if _overlay is None: return
    if self.xxyy is None:
      self.overlay = _overlay
      self.iniLims()
    _yy = [self.xxyy[1][0], self.xxyy[1][1]]
    self.setWave(_overlay)
    if self.plot is None: return
    self.plot.updateRanges([self.xxyy[0], _yy])
    self.setWave() # this shouldn't be needed but is
  def toggleOverlay(self):
    self.setOverlay(not(self.overlay))
  def viewRangeChanged(self, _xxyy):
    self.xxyy = self.plot.xxyy

    # Bypass if called during plot constuction
    if self.plots is None: return
    if self.plots[-1] is None: return

    # Setwave only if calculation required
    if self.mm2.calcMinMax(self.xxyy[0][0], self.xxyy[0][1], self.overlay):
      self.setWave()
    if self.viewRangeChangedFunc is not None:
      self.viewRangeChangedFunc(self, _xxyy)
  def setCursor(self, ev = None, cursorfunc = None):
    self.plot.setCursor(ev, cursorfunc)
  def show(self):
    if self.plot is None: return
    self.plot.show()
  def mouseClickEvent(self, ev):
    if ev.button() == QtLeftButton:
      i = self.mm2.pick(ev.X, ev.Y, self.visual)
      if i is None: return
      ID = self.eid[i]
      if ev.keymod == ABXYKeyModifiers[0]:
        self.Active[2][ID[0]][ID[1]] = True
      else:
        tog = self.Active[2][ID[0]][ID[1]] if self.active[2].sum() == 1 else ev.keymod == ABXYKeyModifiers[1]
        if not(tog):
          self.setActive([None, None, False])
        self.Active[2][ID[0]][ID[1]] = np.logical_not(self.Active[2][ID[0]][ID[1]])
      self.setActive()
      self.setWave()
      #self.setPick()
      ev.sender = self
      ev.action = 2
      self.onActiveChanged(ev)
    elif ev.button() == QtMidButton:
      self.toggleOverlay()
    if self.mouseClickEventFunc is not None:
      ev.action = 1
      ev.sender = self
      self.mouseClickEventFunc(ev)
  def rbEvent(self, ev):
    I = self.mm2.pick(ev.X, ev.Y, self.visual)
    if I is None: return
    if isint(I): I = [I]
    if ev.keymod == ABXYKeyModifiers[1]: # Block-select rather than append
      self.setActive([None, None, False])
    for i in I:
      ID = self.eid[i]
      self.Active[2][ID[0]][ID[1]] = True
    self.setActive()
    self.setWave()
    #self.setPick()
    ev.sender = self
    ev.action = 2
    self.onActiveChanged(ev)
    if self.mouseClickEventFunc is not None:
      ev.action = 1
      ev.sender = self
      self.mouseClickEventFunc(ev)
  def keyPressEvent(self, ev):
    if self.keysDisabled: return
    if ev.key() == QtKeySpace:
      self.toggleOverlay()
    elif ev.key() == QtKeyDelete:
      self.delete()
      self.setWave()
      ev.sender = self
      ev.action = 0
      self.onActiveChanged(ev)
    elif ev.key() in KEY_ALL_INVERT_NONE:
      self.keyUIOmap[ev.key()]()
      self.setWave()
      ev.sender = self
      ev.action = 2
      self.onActiveChanged(ev)
    elif ev.key() == QtKeyInsert:
      self.insert()
      self.setWave()
      ev.sender = self
      ev.action = 0
      self.onActiveChanged(ev)
    elif not(ev.key() in ABXYKeyMod):
      print("Unknown key: ID #:" + str(ev.key()))
    if self.keyPressEventFunc is not None:
      ev.sender = self
      ev.action = 0
      self.keyPressEventFunc(ev)

class pywave2: # two pywaves sharing same data aligned horizontally side-by-side
  pw = None
  ne = 0
  nc = 2
  vr = False
  useDocks = None
  docks = None
  boxes = None
  plotsSet = False
  pwid = None
  ppwid = None
  def __init__(self, _data = [], _chinfo = [], _onsets = [], _Active = [True, True, False]):
    self.setData(_data, _chinfo, _onsets, _Active)
  def setData(self, _data = [], _chinfo = [], _onsets = [], _Active = [True, True, False]):
    if _data is None: return
    if self.pw is None: self.pw = [None] * self.nc
    for i in range(self.nc):
      self.pw[i] = pywave(_data, _chinfo, _onsets, _Active)
    self.chinfo = self.pw[0].chinfo
    self.ne = len(self.pw[0].data)
  def setArea(self, *args, **_kwds):
    self.form = None # inheriting from QMainWindow
    self.boxes = None # GBox() inheriting from QGraphicsLayoutWidget
    self.area = None
    self.docks = None
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
      self.area = area()
      self.parent = self.area
    if self.form is not None:
      self.form.setCentralWidget(self.parent)
  def setPlots(self, *args, **kwds):
    self.setArea(*args, **kwds)
    self.plotsSet = self.nc
    if self.nc is None: return self.parent
    if self.nc == 0: return self.parent
    self.useDocks = self.area is not None
    if self.useDocks:
      self.docks = [None] * self.nc
      self.boxes = [None] * self.nc
    self.pgbar = pgb("Preparing plots ...", self.ne * self.nc)
    for i in range(self.nc):
      if self.useDocks:
        docktitle = "Ch #" + str(i)
        if isinstance(self.chinfo, chWave):
          docktitle = self.chinfo.name + "/" + self.chinfo.units
        self.docks[i] = dock(docktitle)
        self.boxes[i] = gbox()
        self.docks[i].add(self.boxes[i])
        if not(i):
          self.area.add(self.docks[i])
        else:
          self.area.add(self.docks[i], 'right', self.docks[i-1])
        self.pw[i].setPlot(self.pgbar, i, parent = self.boxes[i])
      else:
        self.pw[i].setPlot(self.pgbar, i, parent = self.gbox, row = 1, col = i)
      self.pw[i].setViewRangeChangedFunc(self.viewRangeChanged)
      self.pw[i].setMouseClickEventFunc(self.activeChangedClick)
      self.pw[i].setKeyPressEventFunc(self.activeChangedPress)
      if i: self.pw[i].setOverlay(not(self.pw[i-1].overlay))
      #if not(self.useDocks) and i < self.nc - 1: self.gbox.nextRow()
    self.pgbar.reset()
    return self.parent
  def clrPlots(self):
    if not(self.useDocks):
      if self.pw is not None:
        n = len(self.pw)
        for i in range(n-1, -1, -1):
          self.gbox.remove(self.pw[i].plot)
      return
    self.area.clrDocks(self.docks)
    self.docks = None
    self.boxes = None
    self.plotsSet = 0
  def evid(self, ev):
    for i in range(len(self.pw)):
      if ev.sender == self.pw[i]:
        return i
    return None
  def setFocus(self):
    if not(self.useDocks):
      self.parent.setFocus()
    else:
      self.boxes[0].setFocus()
  def setPens(self, _pens = None, _w = None):
    for _pw in self.pw:
      _pw.setPens(_pens, _w)
  def setLims(self, _Lims = None):
    if _Lims is None: return
    for i in range(len(_Lims)):
      self.pw[i].setLims(_Lims[i])
  def setMarks(self, _Marks = None, _markc = 'T', _markr = 1.0):
    if _Marks is None: _Marks = [None] * self.nc
    for i in range(self.nc):
      self.pw[i].setMarks(_Marks[i], _markc, _markr)
  def viewRangeChanged(self, sender, _xxyy):
    if self.vr: return
    self.vr = True # prevent recursive calls
    k = None
    o = None
    for i in range(self.nc):
      if sender == self.pw[i]:
        k = i
        _xxyy[1] = self.pw[i].plot.xxyy[1]
        o = self.pw[i].overlay
    for i in range(self.nc):
      if i != k:
        if self.pw[i].overlay == o:
          self.pw[i].setOverlay(not(o))
        else:
          xxyyi = [self.pw[i].plot.xxyy[0], _xxyy[1]] # update y only
          self.pw[i].plot.updateRanges(xxyyi)
    self.vr = False
  def activeChanged(self, ev):
    k = self.evid(ev)
    if k is None: return
    active = self.pw[k].cpyActive()
    for i in range(len(self.pw)):
      if i != k:
        self.pw[i].setActive(active)
        self.pw[i].setWave()
  def activeChangedClick(self, ev):
    ev.action = 1
    if ev.button() == QtLeftButton:
      ev.action = 2
    return self.activeChanged(ev)
  def activeChangedPress(self, ev):
    ev.action = 0
    return self.activeChanged(ev)
  def setCursor(self, _pwid = None, ev = None, cursorfunc = None):
    if _pwid is not None:
      self.ppwid = _pwid
    else:
      _pwid = self.ppwid
    self.pwid = _pwid
    self.pw[self.pwid].setCursor(ev, cursorfunc)
  def show(self):
    self.form.show()

class pywav (pywave2): # multiple pywaves aligned vertically with different data
  nr = None
  def __init__(self, _Data = None, _ChInfo = None, _Onsets = None, _Active = [True, True, False]):
    self.setData(_Data, _ChInfo, _Onsets, _Active)
  def setData(self, _Data = None, _ChInfo = None, _Onsets = None, _Active = [True, True, False]):
    if (_Data is None or _ChInfo is None):
      return
    self.Data = _Data
    self.nr = len(self.Data)
    self.ChInfo = repl(_ChInfo, self.nr)
    if _Onsets is None: self.Onsets = [[]] * self.nr
    self.Onsets = repl(_Onsets, self.nr)
    newpw = self.pw is None
    if not(newpw):
      if len(self.pw) != self.nr:
        print("Warning: updated data incommensurate with pre-existing number of channels: clearing data.")
        self.clrData()
        newpw = True
    if newpw: self.pw = []
    for i in range(self.nr):
      if newpw:
        _pw = pywave(self.Data[i], self.ChInfo[i], self.Onsets[i], _Active)
        self.pw.append(_pw)
      else:
        self.pw[i].updateData(self.Data[i], self.ChInfo[i], self.Onsets[i], _Active)
      if not(i):
        self.ne = self.pw[i].ne
        self.ns = self.pw[i].ns
      elif self.ns != self.pw[i].ns or self.ne != self.pw[i].ne:
        raise ValueError("Data sizes for channels incommensurate")
  def setPlots(self, *args, **kwds):
    if self.plotsSet: self.clrPlots()
    self.plotsSet = self.nr
    self.setArea(*args, **kwds)
    if self.nr is None: return self.parent
    if self.nr == 0: return self.parent
    self.useDocks = self.area is not None
    if self.useDocks:
      self.docks = [None] * self.nr
      self.boxes = [None] * self.nr
    self.pgbar = pgb("Preparing plots ...", self.ne * self.nr)
    for i in range(self.nr):
      if self.useDocks:
        docktitle = "Ch #" + str(i)
        if isinstance(self.ChInfo[i], chWave):
          docktitle = self.ChInfo[i].name + "/" + self.ChInfo[i].units
        self.docks[i] = dock(docktitle)
        self.boxes[i] = gbox()
        self.docks[i].add(self.boxes[i])
        self.area.add(self.docks[i])
        self.pw[i].setPlot(self.pgbar, i, parent = self.boxes[i])
      else:
        self.pw[i].setPlot(self.pgbar, i, parent = self.gbox, row = i, col = 1)
      self.pw[i].setViewRangeChangedFunc(self.viewRangeChanged)
      self.pw[i].setMouseClickEventFunc(self.activeChangedClick)
      self.pw[i].setKeyPressEventFunc(self.activeChangedPress)
      if not(self.useDocks) and i < self.nr - 1: self.gbox.nextRow()
    self.pgbar.reset()
    return self.parent
  def updateData(self, *args):
    self.setData(*args)
    if self.nr != self.plotsSet:
      print("Warning: updated data incommensurate with number plots: clearing plots.")
      self.clrPlots()
      return
    for i in range(self.nr):
      self.pw[i].clrGraph()
    self.pgbar = pgb("Preparing plots ...", self.ne * self.nr)
    for i in range(self.nr):
      self.pw[i].setWave(self.pw[i].overlay, self.pgbar, i)
  def clrData(self):
    if self.pw is not None:
      n = len(self.pw)
      for i in range(n-1, -1, -1):
        del self.pw[i]
      del self.pw
      self.pw = None
  def storeLims(self, spec = 2, ind = -1): # (uiid, others) 0 = x, 1 = y, 2 = xy, otherwise not
    if isint(spec): spec = [spec, spec]
    self.Lims = [None] * self.nr
    for i in range(self.nr):
      j = spec[0] if ind == i else spec[1]
      self.Lims[i] = [None, None]
      if j == 0 or j == 2:
        self.Lims[i][0] = self.pw[i].xxyy[0]
      if j == 1 or j == 2:
        self.Lims[i][1] = self.pw[i].xxyy[1]
    return self.Lims
  def resetLims(self, _Lims = None):
    if _Lims is None: _Lims = self.Lims
    for i in range(self.nr):
      self.pw[i].setLims(_Lims[i])
  def viewRangeChanged(self, sender, _xxyy):
    if self.vr: return
    self.vr = True # prevent recursive calls
    k = None
    o = None
    for i in range(self.nr):
      if sender == self.pw[i]:
        k = i
        _xxyy[0] = self.pw[i].plot.xxyy[0]
        o = self.pw[i].overlay
    for i in range(self.nr):
      if i != k:
        if self.pw[i].overlay != o:
          self.pw[i].setOverlay(o)
        else:
          xxyyi = [_xxyy[0], self.pw[i].plot.xxyy[1]] # update x only
          self.pw[i].plot.updateRanges(xxyyi)
    self.vr = False

class pywav2 (pywav): # multiple pywav2s aligned vertically with different data
  np = None
  def __init__(self, _Data = None, _ChInfo = None, _Onsets = None, _Active = [True, True, False]):
    self.setData(_Data, _ChInfo, _Onsets, _Active)
  def setData(self, _Data = None, _ChInfo = None, _Onsets = None, _Active = [True, True, False]):
    if (_Data is None or _ChInfo is None):
      return
    self.Data = _Data
    self.nr = len(self.Data)
    self.nc = 2
    self.np = self.nr * self.nc
    self.ChInfo = repl(_ChInfo, self.nr)
    if _Onsets is None: self.Onsets = [[]] * self.nr
    self.Onsets = repl(_Onsets, self.nr)
    newpw = self.pw is None
    if not(newpw):
      if len(self.pw) != self.nr:
        print("Warning: updated data incommensurate with pre-existing number of channels: clearing data.")
        self.clrData()
        newpw = True
    if newpw: self.pw = []
    h = -1
    for i in range(self.nr):
      for j in range(self.nc):
        h += 1
        if newpw:
          _pw = pywave(self.Data[i], self.ChInfo[i], self.Onsets[i], _Active)
          self.pw.append(_pw)
        else:
          self.pw[h].updateData(self.Data[i], self.ChInfo[i], self.Onsets[i], _Active)
        if not(h):
          self.ne = self.pw[i].ne
          self.ns = self.pw[i].ns
        elif self.ns != self.pw[i].ns or self.ne != self.pw[i].ne:
          raise ValueError("Data sizes for channels incommensurate")
  def setPlots(self, *args, **kwds):
    if self.plotsSet: self.clrPlots()
    self.plotsSet = self.np
    self.setArea(*args, **kwds)
    if self.nr is None or self.nc is None or self.np is None: return self.parent
    if self.np == 0: return self.parent
    self.useDocks = self.area is not None
    if self.useDocks:
      self.docks = [None] * self.np
      self.boxes = [None] * self.np
    self.pgbar = pgb("Preparing plots ...", self.ne * self.np)
    h = -1
    for i in range(self.nr):
      for j in range(self.nc):
        h += 1
        if self.useDocks:
          docktitle = "Ch #" + str(i)
          if isinstance(self.ChInfo[i], chWave):
            docktitle = self.ChInfo[i].name + "/" + self.ChInfo[i].units
          self.docks[h] = dock(docktitle)
          self.boxes[h] = gbox()
          self.docks[h].add(self.boxes[h])
          if not(j):
            self.area.add(self.docks[h])
          else:
            self.area.add(self.docks[h], 'right', self.docks[h-1])
          self.pw[h].setPlot(self.pgbar, float(i)/float(self.nc), parent = self.boxes[h])
        else:
          self.pw[h].setPlot(self.pgbar, float(i)/float(self.nc), parent = self.gbox, row = i, col = j)
        self.pw[h].setViewRangeChangedFunc(self.viewRangeChanged)
        self.pw[h].setMouseClickEventFunc(self.activeChangedClick)
        self.pw[h].setKeyPressEventFunc(self.activeChangedPress)
        if not(h):
          o0 = self.pw[h].overlay
        else:
          _o = not(o0) if j else o0
          self.vr = h < self.np -1
          self.pw[h].setOverlay(_o)
          self.vr = False
      if not(self.useDocks) and i < self.nr - 1: self.gbox.nextRow()
    self.pgbar.reset()
    return self.parent
  def viewRangeChanged(self, sender, _xxyy):
    if self.vr: return
    self.vr = True # prevent recursive calls
    k = None
    o = None
    for i in range(self.np):
      if sender == self.pw[i]:
        k = i
        _xxyy = self.pw[i].plot.xxyy
        o = self.pw[i].overlay
    kr = int(k/self.nc)
    kc = k % 2
    h = -1
    for i in range(self.nr):
      for j in range(self.nc):
        h += 1
        if h != k:
          _o = o if j == kc else not(o)
          if self.pw[h].overlay != _o:
            self.pw[h].setOverlay(_o)
          else:
            xxyyi = None
            if i == kr:
              xxyyi = [self.pw[h].plot.xxyy[0], _xxyy[1]] # update y only
            if j == kc:
              xxyyi = [_xxyy[0], self.pw[h].plot.xxyy[1]] # update x only
            if xxyyi is not None: self.pw[h].plot.updateRanges(xxyyi)
    self.vr = False

class pyscat (xygui):
  defpads = [0.05, 0.05]
  defPens = 'jet'
  defBrushes = 'jet'
  defSizes = 3.
  defSymbols = 'o'
  defEllCol = 'm'
  plot = None
  scat = None
  pick = None
  labels = None
  lblpos = None
  deflblpos = ['bottom', 'left']
  useinside = True
  f = None
  def __init__(self, _X = [], _Y = [], _active = [True, True, False]):
    if DEFSCATMARKERSIZE is not None:
      self.defSizes = DEFSCATMARKERSIZE
    self.setData(_X, _Y, _active)
    self.setActiveChangedFunc()
  def setActiveChangedFunc(self, _activeChangedFunc = None):
    self.activeChangedFunc = _activeChangedFunc
  def setData(self, _X = [], _Y = [], _active = [True, True, False]):
    xygui.setData(self, _X, _Y)
    if _active is None: return
    self.active = _active[:]
    self.setActive(self.active) # Show, Select, Pick
    self.defScat()
    if self.plot is None: return
    # Update scatter to new data set
    self.iniLims()
    self.scat[self.f].setData(self.x, self.y)
    self.pick.setData(self.x, self.y)
    self.setScat(self.f)
  def iniLims(self, _lims = None, _pads = None):
    if _lims is None: _lims = [self.xm, self.ym]
    if _pads is None: _pads = self.defpads
    _xmid = 0.5 * (_lims[0][1] + _lims[0][0])
    _ymid = 0.5 * (_lims[1][1] + _lims[1][0])
    _xwid = 0.5 * (_lims[0][1] - _lims[0][0])
    _ywid = 0.5 * (_lims[1][1] - _lims[1][0])
    _xwid *= (1. + _pads[0])
    _ywid *= (1. + _pads[1])
    self.plot.iniGUI()
    try:
      self.plot.setXRange(_xmid-_xwid, _xmid+_xwid, 0., False)
      self.plot.setYRange(_ymid-_ywid, _ymid+_ywid, 0., True)
    except Exception:
      pass
  def defScat(self):
    if not(self.N): return
    self.jetpens = [None] * self.N
    self.jetbrushes = [None] * self.N
    for i in range(self.N):
      self.jetpens[i] = jetpen(self.n[i])
      self.jetbrushes[i] = jetbrush(self.n[i])
    self.setPens(self.defPens)
    self.setBrushes(self.defBrushes)
    self.setSizes(self.defSizes)
    self.setSymbols(self.defSymbols)
    self.setSelectFuncs()
    self.setPickFuncs()
    self.iniEllipse()
  def iniEllipse(self):
    self.ells = None
    self.ellp = None
  def setEllipse(self, _ellCol = None):
    if _ellCol == False: return
    if _ellCol is None: _ellCol = self.defEllCol
    if self.ells is not None or self.ellp is not None:
      raise ValueError("Cannot set more than one ellipse")
    if self.plot is None:
      raise TypeError("Cannot set ellipse before setting plot")
    self.ellCol = _ellCol
    self.defEllipse() #default points
    self.ellp = pg.PlotCurveItem(self.ellx, self.elly, pen = rgbPen(self.ellCol), brush = rgbBrush(self.ellCol))
    self.ells = self.plot.scat(self.x2, self.y2, pen = rgbPen(self.ellCol), brush = rgbBrush(self.ellCol))
    self.defEllipse() # re-run to test on data
    self.plot.addItem(self.ellp)
    self.plot.addDragFunc(self.dragEllipse)
  def setNumArg(self, ip = None):
    if ip is None: return None
    if nDim(ip) == 0: ip = [ip] * self.N
    op = [None] * self.N
    for i in range(self.N):
      if not(nDim(ip[i])):
        op[i] = np.tile(ip[i], self.n[i])
      else:
        op[i] = ip[i]
    return op
  def setTxtArg(self, ip = None, jetspec = None):
    if ip is None: return None
    if nDim(ip) == 0: ip = [ip] * self.N
    op = [None] * self.N
    for i in range(self.N):
      if ip[i] == 'jet':
        if jetspec is None:
          raise ValueError("Jet specification not matched by specification.")
        op[i] = jetspec[i]
      elif nDim(ip[i]) == 0:
        op[i] = [ip[i]] * self.n[i]
      else:
        op[i] = ip[i]
    return op
  def setPens(self, _pens = None):
    if _pens is None: _pens = self.pens
    self.pens = self.setTxtArg(_pens, self.jetpens)
  def setBrushes(self, _brushes = None):
    if _brushes is None: _brushes = self.brushes
    self.brushes = self.setTxtArg(_brushes, self.jetbrushes)
  def setSizes(self, _sizes = None):
    if _sizes is None: _sizes = self.sizes
    self.sizes = self.setNumArg(_sizes)
  def setSymbols(self, _symbols = None):
    if _symbols is None: _symbols = self.symbols
    self.symbols = self.setTxtArg(_symbols)
  def setPlot(self, *args, **kwds):
    if self.plot is None:
      self.plot = graph(*args, **kwds)
      self.plot.setMouseClickEventFunc(self.mouseClickEvent)
      self.plot.setKeyPressEventFunc(self.keyPressEvent)
    if self.scat is None:
      self.scat = [None] * self.N
      for i in range(self.N):
        self.setFocus(i)
        self.scat[i] = self.plot.scat(self.x, self.y)
        self.setScat(i)
    if self.labels is not None: self.setLabels(self.labels, self.lblpos)
    return self.plot
  def setScat(self, *_args):
    if self.f is None: return
    i = self.f
    h = 1
    _pen = None
    _brush = None
    _size = None
    _symbol = None
    if len(_args):
      if type(_args[0]) is int:
        i = _args[0]
        args = _args[1:]
      else:
        args = _args
      if len(args) > 0: _pen = args[0]
      if len(args) > 1: _brush = args[1]
      if len(args) > 2: _size = args[2]
      if len(args) > 3: _symbol = args[3]
    if _pen is None: _pen = self.pens[i]
    if _brush is None: _brush = self.brushes[i]
    if _size is None: _size = self.sizes[i]
    if _symbol is None: _symbol = self.symbols[i]
    self.setPen(i, pen = _pen)
    self.setBrush(i, brush = _brush)
    self.setSize(i, size = _size)
    self.setSymbol(i, symbol = _symbol)
    self.setPick()
  def setPickFuncs(self, _penfunc = amplify, _brushfunc = None, _sizefunc = enlarge, _symbolfunc = None, _postfunc = None):
    self.pickPenFunc = _penfunc
    self.pickBrushFunc = _brushfunc
    self.pickSizeFunc = _sizefunc
    self.pickSymbolFunc = _symbolfunc
    self.pickPostFunc = _postfunc
  def setSelectFuncs(self, _penfunc = attenuate, _brushfunc = attenuate, _sizefunc = None, _symbolfunc = None, _postfunc = None):
    # def setSelectFuncs(self, _penfunc = attenuate, _brushfunc = attenuate, _sizefunc = shrink, _symbolfunc = None, _postfunc = None):
    self.selectPenFunc = _penfunc
    self.selectBrushFunc = _brushfunc
    self.selectSizeFunc = _sizefunc
    self.selectSymbolFunc = _symbolfunc
    self.selectPostFunc = _postfunc
  def onActiveChanged(self, ev = None):
    if self.activeChangedFunc is None: return
    if ev is not None: ev.sender = self
    self.activeChangedFunc(ev)
  def setActive(self, _active = [None, None, None]):
    # First index: Boolean logic for whether showable (i.e. alive) or not (i.e. dead)
    # Second index: Boolean logic for selected events
    # Third index: Boolean logic for picked events
    if not(self.N): return
    i = self.f
    for i in range(3):
      if type(_active[i]) is bool: _active[i] = np.tile(_active[i], self.nf)
      if _active[i] is not None:
        self.active[i] = _active[i]
    self.active[2] = np.logical_and(self.active[0], self.active[2])
  def setPick(self, rbfunc = True):
    if not(self.N): return None
    i = self.f
    self.pickPen = self.pens[i][:]
    self.pickBrush = self.brushes[i][:]
    self.pickSize = np.copy(self.sizes[i])
    self.pickSymbol = self.symbols[i][:]
    self.pickSize[np.logical_not(self.active[2])] = 0
    if self.pickPenFunc is not None:
      for i in range(self.nf):
        self.pickPen[i] = self.pickPenFunc(self.pickPen[i])
    if self.pickBrushFunc is not None:
      for i in range(self.nf):
        self.pickBrush[i] = self.pickBrushFunc(self.pickBrush[i])
    if self.pickSizeFunc == enlarge:
      for i in range(self.nf):
        self.pickSize[i] = self.pickSizeFunc(self.pickSize[i])
    elif self.pickSizeFunc is not None:
      for i in range(self.nf):
        self.pickSize[i] = self.pickSizeFunc(self.active[2][i])
    if self.pickSymbolFunc is not None:
      for i in range(self.nf):
        self.pickSymbol[i] = self.pickSymbolFunc(self.pickSymbol[i])
    #'''
    if self.pick is None: self.pick = self.plot.scat(self.x, self.y)
    self.pick.setPen(self.pickPen, mask = None)
    self.pick.setBrush(self.pickBrush, mask = None)
    self.pick.setSize(self.pickSize)
    self.pick.setSymbol(self.pickSymbol)
    #'''
    if rbfunc is False: return
    if type(rbfunc) is bool: rbfunc = self.rbEvent
    if not(len(self.plot.rbFunc)): self.plot.addRbFunc(rbfunc)
  def setPen(self, i = None, *args, **kwds):
    if i is None: i = self.f
    if 'pen' in kwds:
      pen = kwds['pen']
      self.pens[i] = pen
      self.setPens(self.pens)
    else:
      pen = self.pens[i]
    pen = pen[:]
    if i == self.f and self.selectPenFunc is not None:
      for k in range(self.nf):
        if not(self.active[0][k]):
          pen[k] = nullpen()
        elif not(self.active[1][k]):
          pen[k] = self.selectPenFunc(pen[k])
    self.scat[i].setPen(pen, *args, mask = None, **kwds)
  def setBrush(self, i = None, *args, **kwds):
    if i is None: i = self.f
    if 'brush' in kwds:
      brush = kwds['brush']
      self.brushes[i] = brush
      self.setBrushes(self.brushes)
    else:
      brush = self.brushes[i]
    brush = brush[:]
    if i == self.f and self.selectBrushFunc is not None:
      for k in range(self.nf):
        if not(self.active[0][k]):
          brush[k] = nullbrush()
        elif not(self.active[1][k]):
          brush[k] = self.selectBrushFunc(brush[k])
    self.scat[i].setBrush(brush, *args, mask = None, **kwds)
  def setSize(self, i = None, *args, **kwds):
    if i is None: i = self.f
    if 'size' in kwds:
      size = kwds['size']
      self.sizes[i] = np.copy(size)
      self.setSizes(self.sizes)
      del kwds['size']
    else:
      size = self.sizes[i]
    size = np.copy(size)
    if i == self.f and self.selectSizeFunc is not None:
      for k in range(self.nf):
        if not(self.active[1][k]):
          size[k] = self.selectSizeFunc(size[k])
    self.scat[i].setSize(size, *args, **kwds)
  def setSymbol(self, i = None, *args, **kwds):
    if i is None: i = self.f
    if 'symbol' in kwds:
      symbol = kwds['symbol']
      self.symbols[i] = symbol
      self.setSymbols(self.symbols)
      del kwds['symbol']
    else:
      symbol = self.symbol[i]
    symbol = symbol[:]
    if i == self.f and self.selectSymbolFunc is not None:
      for k in range(self.nf):
        if not(self.active[1][k]):
          symbol[k] = self.selectSymbolFunc(symbol[k])
    self.scat[i].setSymbol(symbol, *args, **kwds)
  def setLabels(self, xy = None, pos = None, **kwds):
    if type(xy) is tuple: xy=list(xy)
    if type(xy) is not list: xy=[xy]
    if len(xy) == 1: xy = [xy[0], None]
    if self.lblpos is None:
      self.lblpos = self.deflblpos if pos is None else pos
    else:
      if pos is not None: self.lblpos = pos
    self.labels = xy
    if self.plot is None: return
    self.plot.setLabels(self.labels, self.lblpos, **kwds)
  def calcEllipse(self, _w = None, _z = None, _xyabps = None):
    xygui.calcEllipse(self, _w, _z, _xyabps)
    if self.ells is None: return
    self.active[1] = self.inellipse if self.useinside else np.logical_not(self.inellipse)
    self.setActive(self.active)
    if self.plot is not None and not(self.dragtype): self.setScat()
  def dragEllipse(self, ev):
    ellgui = xygui.dragEllipse(self, ev.status, ev.X, ev.Y)
    if ev.status == 1:
      return ellgui
    if ev.status == 2:
      self.ellp.setData(self.ellx, self.elly)
      self.ells.setData(self.x4, self.y4)
      return
    self.ellp.setData(self.ellx, self.elly)
    self.ells.setData(self.x2, self.y2)
    ev.action = 1
    self.onActiveChanged(ev)
  def mouseClickEvent(self, ev):
    if ev.button() == QtLeftButton:
      i = self.argnear(ev.X, ev.Y, True, self.plot.xxyy[0], self.plot.xxyy[1])
      if ev.keymod == ABXYKeyModifiers[0]:
        self.active[2][i] = True
      else:
        tog = self.active[2][i] if self.active[2].sum() == 1 else ev.keymod == ABXYKeyModifiers[1]
        if not(tog):
          self.setActive([None, None, False])
        self.active[2][i] = np.logical_not(self.active[2][i])
      self.setActive()
      self.setScat()
      ev.action = 2
      self.onActiveChanged(ev)
    elif ev.button() == QtRightButton:
      if self.ells is not None:
        self.defEllipse()
        ev.status = 0
        self.dragEllipse(ev) # drag the ellipse back to default position
        ev.action = 1
        self.onActiveChanged(ev)
    elif ev.button() == QtMidButton:
      if self.ells is not None:
        self.useinside = not(self.useinside)
        self.calcEllipse()
        self.setScat()
        ev.status = 0
        ev.action = 1
        self.onActiveChanged(ev)
  def rbEvent(self, ev):
    i = self.argnear(ev.X, ev.Y, True, self.plot.xxyy[0], self.plot.xxyy[1])
    if ev.keymod == ABXYKeyModifiers[1]: # Block-select rather than append
      self.setActive([None, None, False])
    self.active[2][i] = True
    self.setActive()
    self.setScat()
    ev.action = 2
    self.onActiveChanged(ev)
  def keyPressEvent(self, ev):
    if ev.key() == QtKeyEscape:
      print("Escape")
    ev.action = 0
    self.onActiveChanged(ev)
  def show(self):
    if self.plot is None: return
    self.plot.form.show()

