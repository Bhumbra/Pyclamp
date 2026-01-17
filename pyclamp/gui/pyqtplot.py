import os
import numpy as np
import shutil
import pyqtgraph as pg
from pyqtgraph.exporters.ImageExporter import ImageExporter
try:
  from pyqtgraph.exporters.ImageExporter import ImageExporter
except ImportError:
  import pyqtgraph.exporters.ImageExporter as ImageExporter
try:
  from pyqtgraph.opengl.MeshData import MeshData
except ImportError:
  MeshData = None

import subprocess
import PyQt6
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyqtgraph.dockarea import DockArea, Dock
import pyqtgraph.opengl as gl # ImportError means python-opengl is missing
from pyqtgraph.opengl.GLViewWidget import GLViewWidget
from pyqtgraph.opengl import GLSurfacePlotItem
import pyqtgraph.multiprocess as pgmp
from time import time

BaseFormClass = QtWidgets.QMainWindow
BaseGboxClass = pg.GraphicsLayoutWidget
BaseLboxClass = pg.LayoutWidget
BaseBboxClass = QtWidgets.QDialogButtonBox
BaseAreaClass = DockArea
BaseDockClass = Dock
BasePlotClass = pg.PlotItem
BaseTablClass = pg.TableWidget
BaseImagClass = pg.ImageView
BaseHLUTClass = pg.HistogramLUTWidget
BaseVboxClass = GLViewWidget
BaseMeshClass = MeshData
BaseSurfClass = GLSurfacePlotItem
BaseTextClass = pg.TextItem
BaseAnimClass = ImageExporter

QtRBButton = QtCore.Qt.MouseButton.LeftButton
QtNoButton = QtCore.Qt.MouseButton.NoButton
QtCoreQPointF = QtCore.QPointF
QtWidgetsQKeyEvent = QtGui.QKeyEvent
QtWidgetsQMouseEvent = QtGui.QMouseEvent
QtWidgetsQGraphicsMouseEvent = QtWidgets.QGraphicsSceneMouseEvent
KeyboardModifiers = QtWidgets.QApplication.keyboardModifiers
QtShiftModifier = QtCore.Qt.KeyboardModifier.ShiftModifier
QtMetaModifier = QtCore.Qt.KeyboardModifier.MetaModifier
QtControlModifier = QtCore.Qt.KeyboardModifier.ControlModifier
QtAlternateModifier = QtCore.Qt.KeyboardModifier.AltModifier
QtEscapeKey =  16777216 # ESCAPE
QtKeyModifiers = [QtControlModifier, QtShiftModifier, QtMetaModifier, QtAlternateModifier]
QtApply = QtWidgets.QDialogButtonBox.StandardButton.Apply
QtCancel = QtWidgets.QDialogButtonBox.StandardButton.Cancel
# Uncomment the next line to switch between Ctrl/Shift and Meta/Alt
#QtKeyModifiers = [QtMetaModifier, QtAlternateModifier, QtControlModifier, QtShiftModifier]


def cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
  """
  Return a MeshData instance with vertexes and faces computed
  for a cylindrical surface.
  The cylinder may be tapered with different radii at each end (truncated cone)
  """
  verts = np.empty((rows+1, cols, 3), dtype=float)
  if isinstance(radius, int):
      radius = [radius, radius] # convert to list
  ## compute vertexes
  th = np.linspace(2 * np.pi, 0, cols).reshape(1, cols)
  r = np.linspace(radius[0],radius[1],num=rows+1, endpoint=True).reshape(rows+1, 1) # radius as a function of z
  verts[...,2] = np.linspace(0, length, num=rows+1, endpoint=True).reshape(rows+1, 1) # z
  if offset:
      th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1,1)) ## rotate each row by 1/2 column
  verts[...,0] = r * np.cos(th) # x = r cos(th)
  verts[...,1] = r * np.sin(th) # y = r sin(th)
  verts = verts.reshape((rows+1)*cols, 3) # just reshape: no redundant vertices...
  ## compute faces
  faces = np.empty((rows*cols*2, 3), dtype=np.uint)
  rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) + np.array([[0, 0, cols]])
  rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, 0, cols]])
  for row in range(rows):
      start = row * cols * 2
      faces[start:start+cols] = rowtemplate1 + row * cols
      faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
  
  return BaseMeshClass(vertexes=verts, faces=faces)

def truncone(_xyz0 = (0, 0, 0), _xyz1 = (0, 0, 1), _rr = 1, *_args, **kwds):
  args = (1, 40) if not(len(_args)) else  _args[:]
  if len(args) == 1: args = (1, args[0])
  if type(_rr) is float or type(_rr) is int: _rr = [_rr]
  xyz0, xyz1, rr = np.array(_xyz0, dtype = float), np.array(_xyz1, dtype = float), np.array(_rr, dtype = float)
  if len(rr) == 1: rr = np.tile(_rr, 2)
  xyzd = xyz1 - xyz0
  xyzh = np.sqrt(np.sum(xyzd**2.))
  xyzo = np.array((0., 0., xyzh), dtype = float)
  xyzm = 0.5 * (xyzo + xyzd)
  xyzc = cylinder(args[0], args[1], rr, length = xyzh)
  xyzg = gl.GLMeshItem(meshdata = xyzc, **kwds) 
  xyzg.rotate(180, xyzm[0], xyzm[1], xyzm[2])
  xyzg.translate(xyz0[0], xyz0[1], xyz0[2])
  #xyzg.setColor((R, G, B, A))
  if 'color' in kwds: xyzg.setColor(kwds['color'])
  return xyzg

class area (BaseAreaClass):
  Docks = None
  def __init__(self, *args, **_kwds):
    self.form = None
    kwds = dict(_kwds)
    if 'form' in kwds:
      self.form = kwds['form']
      del kwds['form']
    else:
      pass
    BaseAreaClass.__init__(self, *args, **kwds)
    if self.form is not None:
      self.form.setCentralWidget(self)
  def add(self, *args, **kwds):
    if self.Docks is None: self.Docks = []
    newdock=False
    _args = list(args)
    if len(args):
      if isinstance(_args[0], dock) or isinstance(_args[0], BaseAreaClass):
        self.addDock(*args, **kwds) 
        self.Docks.append(_args[0])
        return _args[0]
    if len(args):
      if type(_args[0]) is str:
        _dock = dock(_args[0])
        self.Docks.append(_dock)
        self.addDock(*args[1:], **kwds)
        return _dock
    _dock = dock()
    self.Docks.append(_dock)
    self.addDock(*args, **kwds)
    return _dock
  def clrDocks(self, _Docks = None):
    if self.Docks is None: self.Docks = []
    if _Docks is None: _Docks = self.Docks[:]
    _Docks = _Docks[:] if type(_Docks) is list else [_Docks]
    n = len(self.Docks)
    if not(len(_Docks)) or not(n):
      return []
    I = []
    for i in range(n-1, -1, -1):
      match = False
      for _dock in _Docks:
        if not(match):
          match = _dock == i if type(_dock) is int else _dock == self.Docks[i]
          if match:
            I.append(i)
            self.Docks[i].clrBoxes()
            self.Docks[i].remove()
            del self.Docks[i]
    if not(len(self.Docks)):
      del self.Docks
      self.Docks = None
    return I
  def remove(self):
    self.close()

class dock (BaseDockClass):
  Boxes = None
  def __init__(self, *args, **kwds):
    if not(len(args)) and not(len(kwds)):
      BaseDockClass.__init__(self, "", size=(1,1))
      return
    BaseDockClass.__init__(self, *args, **kwds)
  def add(self, *args, **kwds):
    if not(len(args)): return
    if self.Boxes is None: self.Boxes = []
    self.addWidget(*args, **kwds)
    self.Boxes.append(list(args)[0])
    return self.Boxes[-1]
  def addBbox(self, *args, **kwds):
    _box = bbox(*args, **kwds)
    return self.add(_box)
  def addGbox(self, *args, **kwds):
    _box = gbox(*args, **kwds)
    return self.add(_box)
  def addVbox(self, *args, **kwds):
    _box = vbox(*args, **kwds)
    return self.add(_box)
  def addLbox(self, *args, **kwds):
    _box = lbox(*args, **kwds)
    return self.add(_box)
  def clrBoxes(self, _Boxes = None):
    if self.Boxes is None: self.Boxes = []
    if _Boxes is None: _Boxes = self.Boxes[:]
    _Boxes = _Boxes[:] if type(_Boxes) is list else [_Boxes]
    n = len(self.Boxes)
    if not(len(_Boxes)) or not(n):
      return []
    I = []
    for i in range(n-1, -1, -1):
      match = False
      for _box in _Boxes:
        if not(match):
          match = _box == i if type(_box) is int else _box == self.Boxes[i]
          if match:
            I.append(i)
            self.Boxes[i].clear()
            del self.Boxes[i]
    if not(len(self.Boxes)):
      del self.Boxes
      self.Boxes = None
    return I
  def remove(self):
    self.close()

class bbox (BaseBboxClass): # A button box widget
  def __init__(self, *args, **kwds):
    BaseBboxClass.__init__(self, *args, **kwds)
    self.Buttons = []
  def addButton(self, *args, **kwds):
    if not(len(args)):
      self.Buttons.append(BaseBboxClass.addButton(self, QtApply, **kwds))
      self.Buttons[-1].setIconSize(QtCore.QSize(1, 1))
      self.Buttons[-1].setText("")
    else:  
      self.Buttons.append(BaseBboxClass.addButton(self, *args, **kwds))
    return self.Buttons[-1]
  def Connect(self, i, connection):
    self.Buttons[i].clicked.connect(connection)
  def setText(self, i, text):
    self.Buttons[i].setText(str(text))
  def setCols(self, i, hicol = None, locol = None, fgcol = 'white'):
    if locol is None: locol = hicol
    self.Buttons[i].setStyleSheet('QPushButton {color: '+ str(fgcol) + ';\
      background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 rgba'
      +str(tuple(locol))+', stop:1 rgba'+str(tuple(hicol))+');}')
  def setIconSize(self, i, _iconSize):
    self.Buttons[i].setIconSize(_iconSize)
  def setWidth(self, i, _width):
    self.Buttons[i].setMinimumWidth(_width)
    self.Buttons[i].setMaximumWidth(_width)

class gbox (BaseGboxClass): # A graphics box widget with convenient animation functionality
  def __init__(self, *args, **_kwds):
    self.form = None
    kwds = dict(_kwds)
    if 'form' in kwds:
      self.form = kwds['form']
      del kwds['form']
    else:
      pass
    BaseGboxClass.__init__(self, *args, **kwds)
    if self.form is not None:
      self.form.setCentralWidget(self)
    self.setKeyPressEventFunc()
    self.setKeyReleaseEventFunc()
  def setKeyPressEventFunc(self, _keyPressEventFunc = None):
    self.keyPressEventFunc = _keyPressEventFunc
  def setKeyReleaseEventFunc(self, _keyReleaseEventFunc = None):
    self.keyReleaseEventFunc = _keyReleaseEventFunc
  def keyPressEvent(self, ev):
    BaseGboxClass.keyPressEvent(self, ev)
    if self.keyPressEventFunc is not None:
      self.keyPressEventFunc(ev)
  def keyReleaseEvent(self, ev):
    BaseGboxClass.keyReleaseEvent(self, ev)
    if self.keyReleaseEventFunc is not None:
      self.keyReleaseEventFunc(ev)
  def add(self, *args, **kwds):
    self.addItem(*args, **kwds)
  def remove(self, *args, **kwds):
    self.removeItem(*args, **kwds)

class vbox (BaseVboxClass): # A OpenGL view box widget with conventient animation functionality
  def __init__(self, *args, **kwds):
    BaseVboxClass.__init__(self, *args, **kwds)
  def add(self, *args, **kwds):
    self.addItem(*args, **kwds)

class lbox (BaseLboxClass): # A layout widget
  def __init__(self, *args, **_kwds):
    self.form = None
    kwds = dict(_kwds)
    if 'form' in kwds:
      self.form = kwds['form']
      del kwds['form']
    else:
      pass
    BaseLboxClass.__init__(self, *args, **kwds)
    if self.form is not None:
      self.form.setCentralWidget(self)
    self.setKeyPressEventFunc()
    self.setKeyReleaseEventFunc()
  def setKeyPressEventFunc(self, _keyPressEventFunc = None):
    self.keyPressEventFunc = _keyPressEventFunc
  def setKeyReleaseEventFunc(self, _keyReleaseEventFunc = None):
    self.keyReleaseEventFunc = _keyReleaseEventFunc
  def keyPressEvent(self, ev):
    BaseLboxClass.keyPressEvent(self, ev)
    if self.keyPressEventFunc is not None:
      self.keyPressEventFunc(ev)
  def keyReleaseEvent(self, ev):
    BaseLboxClass.keyReleaseEvent(self, ev)
    if self.keyReleaseEventFunc is not None:
      self.keyReleaseEventFunc(ev)
  def add(self, *args, **kwds):
    self.addWidget(*args, **kwds)

class anim (BaseAnimClass): # animates a gbox
  animfunc = None
  patn = ''
  abox = None
  form = None
  adim = 0
  defwidth = 960
  defheight = 600
  deffps = 30
  def __init__(self, *args, **kwds):
    self.initialise(*args, **kwds)
    self.setRes()
  def initialise(self, *args, **kwds):
    if not(len(args)):
      return
    else:
      _args = list(args)
      self.setAbox(_args[0])
    if self.adim:
      if self.adim == 2:
        _args[0] = _args[0].scene()
        args = tuple(_args)
        BaseAnimClass.__init__(self, *args, **kwds)
  def setAbox(self, _abox = None):
    self.adim = 0
    self.abox = _abox
    if self.abox is None: return
    if isinstance(self.abox, gbox): 
      self.adim = 2
    elif isinstance(self.abox, vbox):
      self.adim = 3
    self.form = self.abox.parent()
    while not(isinstance(self.form, BaseFormClass)):
      self.form = self.form.parent()
  def setAnimFunc(self, func, *args, **kwds):
    self.animfunc = func
    self.animargs = args
    self.animkwds = kwds
  def setParam(self, **kwds):
    if self.adim != 2: 
      print("Warning: anim.setParam() only effective for 2D graphics boxes")
      return
    if not(len(kwds)): return
    keys = []
    vals = []
    for key, val in kwds.iteritems():
      keys = "%s" % key
      self.parameters()[keys] = val
  def setRes(self, _width = None, _height = None, _fps = None):
    self.width = self.defwidth if _width is None else _width
    self.height = self.defheight if _height is None else _height
    self.fps = self.deffps if _fps is None else _fps
  def saveFrame(self, fn):
    if self.adim == 2:
      self.export(fn)
    elif self.adim == 3:
      self.abox.readQImage().save(fn)
    else:
      raise ValueError("Frame saving not supported for box type")
  def animate(self, n, _patn = None, _pgb = None):
    self.form.resize(self.width, self.height)
    self.abox.resize(self.width, self.height)
    if self.adim == 2:
      self.parameters()['width'] = self.width
      self.parameters()['height'] = self.height
    if _patn is None:
     _patn = raw_input("Select path for animation (Enter to not save): ")
    self.patn = _patn
    dirn, filn = os.path.split(self.patn)
    self.stem, self.outextn = os.path.splitext(filn)
    if not(len(dirn)): dirn = os.getcwd()
    self.outpath = dirn + '/' + self.stem
    self.pngdirn = dirn + '/' + "." + self.stem
    if len(self.patn):
      if os.path.exists(self.pngdirn):
        print("Warning: anim.animate(): undeleted cache directory detected: " + self.pngdirn)
      else:
        os.makedirs(self.pngdirn)
    if _pgb is not None: _pgb.init('Rendering frames', n)
    self.N = len(str(n))
    for i in range(n):
      if _pgb is not None: _pgb.set(i)
      j = str(i)
      k = len(j)
      if k < self.N: j = (self.N-k) * '0' + j
      if self.animfunc is not None: self.animfunc(*self.animargs, **self.animkwds)
      if len(self.patn):
        self.saveFrame(self.pngdirn + '/' + self.stem + "_" + j + '.png')
    if _pgb is not None: _pgb.reset()
    return self
  def saveAnim(self, keeppng = False, enc = None):
    if not(len(self.patn)): return
    if enc == "mencoder" or enc == "avi":
      if not(len(self.outextn)): self.outextn = ".avi"
      pngstr = self.pngdirn + '/' + "*.png"
      encstr =  'mencoder mf://' + pngstr 
      encstr += ' -mf ' + 'w=' + str(self.width) + ':h=' + str(self.height) + ":fps=" + str(self.fps)
      encstr += ":type=png -o " + self.outpath + self.outextn + " -oac copy -ovc copy "
      #encstr += '-oac lavc -ovc lavc '
      #encstr += '-oac copy -ovc x264 '
      #encstr += "-of lavf -oac lavc -lavcopts acodec=wmav2 -ovc lavc -lavcopts vcodec=wmv2"
      #encstr += "-of lavf -oac lavc -lavcopts acodec=wmav1 -ovc lavc -lavcopts vcodec=wmv1"
      #encstr += "-of lavf -oac lavc -ovc lavc -lavcopts vcodec=wmv1:vbitrate=1500:acodec=wmav1"
      #encstr += "-of mpeg -oac mp3lame -ovc lavc -lavcopts vcodec=mpeg1video" 
    elif enc == "ffmpeg" or enc == "swf":
      if not(len(self.outextn)): self.outextn = ".swf"
      pngstr = self.pngdirn + '/' + self.stem + "_%0" + str(self.N) + "d.png"
      encstr = "ffmpeg -framerate " + str(self.fps) + " -i " + pngstr
      encstr += " -q:v 0 -q:a 0 " + self.outpath + self.outextn + " "
    else:
      pngstr = self.pngdirn + '/' + "*.png"
      if not(len(self.outextn)): self.outextn = ".gif"
      encstr =  'convert -delay 1x' + str(self.fps) + " -layers optimize " + pngstr + " " + self.outpath + self.outextn + " "
    os.system(encstr)
    if keeppng: return encstr
    shutil.rmtree(self.pngdirn)
    return encstr

class graph (BasePlotClass): # adds bounding and RB zooming to Base Plot Class
  button = QtNoButton
  rbButton = QtRBButton  
  keyMod = QtKeyModifiers
  xyKeyMod = [QtKeyModifiers[2], QtKeyModifiers[3], QtKeyModifiers[2] | QtKeyModifiers[3]]
  escKey = QtEscapeKey
  defRbPen = pg.functions.mkPen((127, 127, 127), width = 1)
  defRbBrush = pg.functions.mkBrush((127, 127, 127, 63))
  scenenudge = [-4, -4] # scene nudge factor
  Cursors = None
  icursor = None
  ncursor = 0
  _GUI = None # previous GUI
  GUI = None  # present GUI
  cursorfunc = None
  dragfunc = None
  gbox = None
  mkey = None 
  RB = None
  labels = None
  lblpos = None
  deflblpos = ['bottom', 'left']
  def __init__(self, *args, **_kwds):
    self.form = None # QMainWindow
    self.gbox = None # GraphicsLayoutWidget
    self.view = None # ViewBox
    add2gbox = False
    if 'viewBox' in _kwds:
      self.view = _kwds['viewBox']
      self.childGroup = self.view.childGroup
    else:
      kwds = dict(_kwds)
      add2gbox = True
      if 'parent' in kwds:
        self.gbox = kwds['parent']
        self.form = self.gbox.parent()
        del kwds['parent']
      else:
        self.form = BaseFormClass()
        self.gbox = gbox()
        self.form.setCentralWidget(self.gbox)
    BasePlotClass.__init__(self, *args, **kwds)  
    if add2gbox: 
      self.gbox.add(self)
      self.gbox.setKeyPressEventFunc(self.keyPressEvent)
      self.gbox.setKeyReleaseEventFunc(self.keyReleaseEvent)
    if self.view is None:
      self.view = self.getViewBox()
      self.childGroup = self.view.childGroup
    self.sigRangeChanged.connect(self.viewRangeChanged)          # adds PlotWidget functionality here
    #self.view.scene().sigMouseMoved.connect(self.mouseMoveEvent) # adds cursor functionality 
    self.proxy = pg.SignalProxy(self.view.scene().sigMouseMoved, rateLimit=50, slot=self.mouseMoveEvent)
    self.iniGUIFunc()
    self.iniGUI()
    #self.setGUI()
  def show(self):
    if self.form is not None: return self.form.show()
    if self.gbox is not None: return self.gbox.show()
    if self.view is not None: return self.view.show()
    return self.show()
  def setViewRangeChangedFunc(self, _viewRangeChangedFunc = None):
    self.viewRangeChangedFunc = _viewRangeChangedFunc
  def setMouseClickEventFunc(self, _mouseClickEventFunc = None):
    self.mouseClickEventFunc = _mouseClickEventFunc
  def setMouseDoubleClickEventFunc(self, _mouseClickEventFunc = None):
    self.mouseDoubleClickEventFunc = _mouseClickEventFunc
  def setKeyPressEventFunc(self, _keyPressEventFunc = None):
    self.keyPressEventFunc = _keyPressEventFunc
  def iniGUI(self):  
    self.iniLims()
    if self.RB is not None: del self.RB
    self.moved = False
    self.XY0 = None
    self.XY1 = None
    self.RB = None
    self.lockxy = False
    self.setCursor()
    self.setRB()
    self.setGUI()
  def iniGUIFunc(self):
    self.dragFunc = []
    self.rbFunc = []
    self.setViewRangeChangedFunc()
    self.setMouseClickEventFunc()
    self.setMouseDoubleClickEventFunc()
    self.setKeyPressEventFunc()
  def iniLims(self, _xxyy = None):
    if _xxyy is None:
      self.xxyy = _xxyy
      self.XXYY = _xxyy
      return
    self.xxyy = [[_xxyy[0][0], _xxyy[0][1]], [_xxyy[1][0], _xxyy[1][1]]]
    self.XXYY = [[_xxyy[0][0], _xxyy[0][1]], [_xxyy[1][0], _xxyy[1][1]]]
  def setBackground(self, bg):
    self.view.state['background'] = bg
    self.view.updateBackground()
  def setGUI(self, _GUI = True, stGUI = False, **_kwds):
    if stGUI: # cache previous GUI
      self._GUI = self.GUI
    self.GUI = _GUI # True = on, False = default, None = off
  def add(self, *args, **kwds):
    self.addItem(*args, **kwds)
  def remove(self, *args, **kwds):
    self.removeItem(*args, **kwds)
  def addDragFunc(self, _dragfunc = None):
    if _dragfunc is not None:
      self.dragFunc.append(_dragfunc)
  def addRbFunc(self, _rbfunc = None):
    if _rbfunc is not None:
      self.rbFunc.append(_rbfunc)
  def line(self, *args, **kwds):
    self.line = pg.PlotCurveItem(*args, **kwds)
    self.view.addItem(self.line)
    return self.line
  def scat(self, *args, **kwds):
    self.scatter = pg.ScatterPlotItem(*args, **kwds)
    self.view.addItem(self.scatter)
    return self.scatter
  def bar(self, X, Y, **kwds):
    nX, nY = len(X), len(Y)
    centre = nX == nY
    widths = np.diff(X).ravel()
    if centre and nX > 1:
      widths = np.hstack((widths, widths[-1]))
    if centre:
      self.barchart = pg.BarGraphItem(x=X, height=Y, width=widths, **kwds)
    else:
      self.barchart = pg.BarGraphItem(x0=X, height=Y, width=widths, **kwds)
    self.view.addItem(self.barchart)
    return self.barchart
  def mapxy(self, xy, _y = None, rescale = True, fromscene = False):
    if _y is not None: 
      x, y = xy, _y
    elif type(xy) is tuple:
      x, y = xy
    elif type(xy) is QtCoreQPointF:
      x, y = xy.x(), xy.y()
    else:
      x, y, = xy.pos().x(), xy.pos().y()
    if fromscene:
      x += self.scenenudge[0]
      y += self.scenenudge[1]
    X = x - self.view.pos().x()
    Y = y - self.view.pos().y()
    if rescale:
      dx = max(1, self.view.width())
      dy = max(1, self.view.height())
      Y = dy - Y
      X =  self.xxyy[0][0] + (self.xxyy[0][1] - self.xxyy[0][0]) * X / float(dx) 
      Y =  self.xxyy[1][0] + (self.xxyy[1][1] - self.xxyy[1][0]) * Y / float(dy) 
    return X, Y
  def setLabels(self, xy = None, pos = None, **kwds): # keywords: fontSize, color, tickFontSize
    if type(xy) is tuple: xy=list(xy)
    if type(xy) is not list: xy=[xy]
    if len(xy) == 1: xy = [xy[0], None]
    if self.lblpos is None:
      self.lblpos = self.deflblpos if pos is None else pos
    else:
      if pos is not None: self.lblpos = pos
    self.labels = xy
    if self.labels is not None:
      labelStyle = {}
      if 'color' in kwds: labelStyle['color'] = str(kwds['color'])
      if 'fontSize' in kwds: labelStyle['font-size'] = str(kwds['fontSize'])+'Px'
      if self.labels[0] is not None: self.setLabel(self.lblpos[0], self.labels[0], **labelStyle)
      if self.labels[1] is not None: self.setLabel(self.lblpos[1], self.labels[1], **labelStyle)
    if 'tickFontSize' in kwds:
      tickFontSize = kwds['tickFontSize']
      tickFont = QtWidgets.QFont()
      tickFont.setPixelSize(tickFontSize)
      self.getAxis(self.lblpos[0]).setTickFont(tickFont)
      self.getAxis(self.lblpos[1]).setTickFont(tickFont)
      self.getAxis(self.lblpos[0]).setHeight(int(float(tickFontSize) * 2 + 5))
      self.getAxis(self.lblpos[1]).setWidth(int(float(tickFontSize) * 3 + 8))
  def setCursor(self, ev = None, _cursorfunc = None):
    resetCursors = ev is None
    if isinstance(ev, QtWidgetsQKeyEvent): # deal with escaped cursors
      if ev.key() == self.escKey:
        ev.cursors = None
        if self.cursorfunc is not None:
          self.cursorfunc(ev)
        self.moved = True # suppresses unwanted mouseClick events
        self.setGUI(self._GUI)
        resetCursors = True
    if resetCursors:
      _redraw = False
      if self.Cursors is not None: # removing previous cursors if present
        for i in range(len(self.Cursors))[::-1]:
          if self.Cursors[i] is not None:
            for j in range(len(self.Cursors[i]))[::-1]:
              if self.Cursors[i][j] is not None: 
                self.remove(self.Cursors[i][j])
                del self.Cursors[i][j]               # so let's try object destruction rather than removal
                _redraw = True
      self.cursor = None         # cursor specification
      self.Cursors = None        # stored cursor objects
      self.Cursor = None         # current cursor
      self.cursors = None        # list of cursor locations
      self.ncursor = 0           # number of cursors
      self.icursor = None        # index of current cursor
      self.cursorfunc = None     # function to call at the end of the cursors
      if _redraw:
        _lockxy = self.lockxy
        self.lockxy = True
        self.viewRangeChanged(None, None)
        self.lockxy = _lockxy
      return
    if type(ev) is int: ev = [ev] #0 = vertical, #1 = horizontal, #2 = both
    if type(ev) is list:
      if self.ncursor:
        raise ValueError("Cursor already in operation.")
      self.cursorfunc = _cursorfunc
      self.setGUI(False, True) # remember GUI setting and temporarily disable it
      self.cursor = np.array(ev, dtype = int)
      self.ncursor = len(self.cursor)
      self.icursor = -1
      self.Cursors = [None] * self.ncursor
      self.cursors = [None] * self.ncursor
    xpos, ypos = None, None
    if self.icursor >= 0:
      self.Cursor = self.Cursors[self.icursor]
      curs = self.cursor[self.icursor]
      if self.Cursor[0] is not None: xpos = self.Cursor[0].x()
      if self.Cursor[1] is not None: ypos = self.Cursor[1].y()
      if curs == 0:
        self.cursors[self.icursor] = xpos
      elif curs == 1:
        self.cursors[self.icursor] = ypos
      else:
        self.cursors[self.icursor] = (xpos, ypos)
    self.icursor += 1
    if self.icursor < self.ncursor:
      i = self.icursor
      curs = self.cursor[i]
      self.Cursors[i] = [None, None]
      if curs == 0 or curs == 2:
        if xpos is None:
          self.Cursors[i][0] = pg.InfiniteLine(angle=90, movable = False)
        else:  
          self.Cursors[i][0] = pg.InfiniteLine(pos=xpos, angle=90, movable=False)
        self.add(self.Cursors[i][0], ignoreBounds=True)
      if curs == 1 or curs == 2:
        if ypos is None:
          self.Cursors[i][1] = pg.InfiniteLine(angle=0,  movable=False)
        else:
          self.Cursors[i][1] = pg.InfiniteLine(pos=ypos, angle=0,  movable=False)
        self.add(self.Cursors[i][1], ignoreBounds=True)
      self.Cursor = self.Cursors[i]
      if self.gbox is not None: self.gbox.setFocus() # move focus to Graphics Box is possible
    else:    
      ev.cursors = self.cursors[:]
      if self.cursorfunc is not None:
        self.cursorfunc(ev)
      self.moved = True      # suppresses unwanted mouseClick events
      self.setGUI(self._GUI) # restore previous GUI
  def setRB(self, ev = None, pen = None, brush = None):
    if pen is not None and brush is not None:
      if self.RB is not None:
        self.RB.hide()
        self.RB = None
    if self.RB is None:
      if pen is None: pen = self.defRbPen
      if brush is None: brush = self.defRbBrush 
      self.RB = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
      self.RB.setPen(pen)
      self.RB.setBrush(brush)
      self.RB.hide()    
      if ev is None: self.view.addItem(self.RB, ignoreBounds=True)
      return
    elif ev is None: # release mouse button  
        self.RB.hide()
        return
    if type(ev) is QtWidgetsQGraphicsMouseEvent:  
      self.XY1 = ev.pos()
    self.keymod = ev.keymod
    x0, y0 = self.XY0.x(), self.XY0.y()
    x1, y1 = self.XY1.x(), self.XY1.y()
    xY, yX = False, False
    xr, yr = x0 > x1, y0 > y1 # Note: y are pixel values from top(left)
    if self.keymod not in self.keyMod:
      xY, yX = xr, yr
    else:
      xY = self.keymod == self.xyKeyMod[0]
      yX = self.keymod == self.xyKeyMod[1]
    if xY:
      x0 = 0.
      x1 = self.width()
    if yX:
      y0 = 0.
      y1 = self.height()
    X0, Y0 = self.mapxy(x0, y0, False) # bypasses axis
    X1, Y1 = self.mapxy(x1, y1, False) # transformation
    XY0 = QtCore.QPointF(X0, Y0)
    XY1 = QtCore.QPointF(X1, Y1)
    try: # to cope with old Qt4 versions.
      rb = QtCore.QRectF(XY0, XY1)
    except TypeError:
      XY0 = QtCore.QPointF(X0, Y0)
      XY1 = QtCore.QPointF(X1, Y1)
      rb = QtCore.QRectF(XY0, XY1)
    rb = self.childGroup.mapRectFromParent(rb)
    #self.RB.setPos(rb.topLeft())
    #self.RB.resetTransform()
    #self.RB.scale(rb.width(), rb.height())
    self.RB.setRect(rb)
    self.RB.resetTransform()
    self.RB.show()
  def rbZoom(self):
    _xxyy = [[self.xxyy[0][0], self.xxyy[0][1]], [self.xxyy[1][0], self.xxyy[1][1]]]
    x0, y0 = self.mapxy(self.XY0) # now x0 and y0 are graph
    x1, y1 = self.mapxy(self.XY1) # co-ordinates
    xY, yX = False, False
    xr, yr = x0 > x1, y0 < y1
    if self.keymod not in self.keyMod:
      xY, yX = xr, yr
    else:
      xY = self.keymod == self.xyKeyMod[1]
      yX = self.keymod == self.xyKeyMod[0]
      if not(xY) and not(yX): return False
    if xY and yX: 
      self.unZoom()
      return True
    if not(xY):
      _xxyy[0][0] = min(x0, x1)
      _xxyy[0][1] = max(x0, x1)
    if not(yX):
      _xxyy[1][0] = min(y0, y1)     
      _xxyy[1][1] = max(y0, y1)
    self.updateRanges(_xxyy)
    return True
  def unZoom(self, spec = 2):
    _xxyy = [[self.xxyy[0][0], self.xxyy[0][1]], [self.xxyy[1][0], self.xxyy[1][1]]]
    if spec == 0 or spec == 2:
      _xxyy[0] = [self.XXYY[0][0], self.XXYY[0][1]]
    if spec == 1 or spec == 2:
      _xxyy[1] = [self.XXYY[1][0], self.XXYY[1][1]]
    self.updateRanges(_xxyy)
  def firstEvent(self, ev):
    if self.XXYY is not None: return
    if self.xxyy is None: self.xxyy = self.viewRange() 
    self.XXYY = [[self.xxyy[0][0], self.xxyy[0][1]], [self.xxyy[1][0], self.xxyy[1][1]]]
  def bound(self, _xxyy):
    self.keymod = KeyboardModifiers()
    self.boundx = False
    self.boundy = False
    if self.XXYY is None: return _xxyy
    xx, yy = _xxyy[0], _xxyy[1]
    xx = [min(xx[0], xx[1]), max(xx[0], xx[1])]
    yy = [min(yy[0], yy[1]), max(yy[0], yy[1])]
    if self.keymod == self.xyKeyMod[1]:
      xx = self.xxyy[0]
      self.boundx = True
    else:
      xx[0] = max(self.XXYY[0][0], xx[0])
      xx[1] = min(self.XXYY[0][1], xx[1])
      self.boundx = xx[0] <= self.XXYY[0][0] or xx[1] >= self.XXYY[0][1]
    if self.keymod == self.xyKeyMod[0]:
      yy = self.xxyy[1]
      self.boundy = True
    else:  
      yy[0] = max(self.XXYY[1][0], yy[0])
      yy[1] = min(self.XXYY[1][1], yy[1])
      self.boundy = yy[0] <= self.XXYY[1][0] or yy[1] >= self.XXYY[1][1]
    return [xx, yy]
  def updateRanges(self, _xxyy = None):
    if  _xxyy is not None: self.xxyy = [[_xxyy[0][0], _xxyy[0][1]], [_xxyy[1][0], _xxyy[1][1]]]
    self.lockxy = True
    self.setXRange(self.xxyy[0][0], self.xxyy[0][1], 0., True) 
    self.setYRange(self.xxyy[1][0], self.xxyy[1][1], 0., True)
    self.lockxy = False
    if self.viewRangeChangedFunc is not None:
      self.viewRangeChangedFunc(self.xxyy)
  def mousePressEvent(self, ev):
    if not(self.GUI):
      if self.GUI is not None:
        if self.Cursor is not None:
          if type(ev) is tuple: ev = ev[0]
          ev.X, ev.Y = self.mapxy(ev, None, True, True) # unusual call
          self.setCursor(ev)
      return  
    self.firstEvent(ev)
    ev.keymod = KeyboardModifiers()
    self.moved = False
    self.dragfunc = None
    for _dragFunc in self.dragFunc:
      ev.status = 1 # 0 = button released, 1 = button down, 2 = drag
      ev.X, ev.Y = self.mapxy(ev)
      if _dragFunc(ev):
        self.button = ev.button()
        self.dragfunc = _dragFunc
    if self.dragfunc is not None: return  
    self.rbButton = QtNoButton if self.view.state['mouseMode'] == pg.ViewBox.RectMode else QtRBButton
    self.button = ev.button()
    if self.button != self.rbButton:
      BasePlotClass.mousePressEvent(self, ev)
      return
    self.XY0 = ev.pos()
  def mouseMoveEvent(self, ev): 
    # Button-independent handing
    if not(self.GUI):
      if self.GUI is not None:
        if self.Cursor is not None:
          if type(ev) is tuple: ev = ev[0]
          ev.X, ev.Y = self.mapxy(ev, None, True, True) # unusual call
          if self.Cursor[0] is not None:
            self.Cursor[0].setPos(ev.X)
          if self.Cursor[1] is not None:
            self.Cursor[1].setPos(ev.Y)
        pass
      return  
    if type(ev) is tuple: # I don't know why some events are tuples
      return 
    # Button-dependent handling
    if self.button == QtNoButton:
      return
    if self.dragfunc is not None:
      self.moved = True
      ev.status = 2
      ev.X, ev.Y = self.mapxy(ev)
      self.dragfunc(ev)
      return
    if self.button != self.rbButton:
      try:
        BasePlotClass.mouseMoveEvent(self, ev)
      except TypeError:
        pass
      self.moved = True
      return
    self.mouseDragEvent(ev)
    self.moved = True    
  def mouseDragEvent(self, ev):
    if not(self.GUI):
      if self.GUI is not None:
        BasePlotClass.mouseDragEvent(self, ev)
      return  
    if type(ev) is tuple: return # I don't know why some events are tuples
    if not(hasattr(ev, 'keymod')):
      ev.keymod = KeyboardModifiers()
    if self.button == self.rbButton:
      self.setRB(ev)
      return
    try:
      BasePlotClass.mouseMoveEvent(self, ev)
    except TypeError:
      pass
  def mouseReleaseEvent(self, ev):
    if not(self.GUI):
      if self.GUI is not None:
        self.view.mouseReleaseEvent(ev)
      return  
    ev.keymod = KeyboardModifiers()
    if self.dragfunc is not None:
      ev.status = 0
      ev.X, ev.Y = self.mapxy(ev)
      self.dragfunc(ev)
      self.button = QtNoButton
      self.dragfunc = None
      if self.moved: 
        self.moved = False
        return
    _moved = self.moved
    self.moved = False
    if self.button != self.rbButton:
      BasePlotClass.mouseReleaseEvent(self, ev)
    elif _moved:
      self.setRB()  # Remove rubber band
      if not(self.rbZoom()): # Zoom 
        for _rbFunc in self.rbFunc:
          x0, y0 = self.mapxy(self.XY0) # now x0 and y0 are graph
          x1, y1 = self.mapxy(self.XY1) # co-ordinates
          ev.X, ev.Y = [x0, x1], [y0, y1]
          _rbFunc(ev)
    if not(_moved):
      self.mouseClickEvent(ev)
    self.button = QtNoButton
  def mouseClickEvent(self, ev): # still click event
    if not(self.GUI):
      if self.GUI is not None:
        BasePlotClass.mousePressEvent.mouseClickEvent(self, ev)
      return  
    if self.mouseClickEventFunc is not None:
      try:
        ev.X, ev.Y = self.mapxy(ev)
      except TypeError: # in case the event fizzles
        return
      self.sender = self
      self.mouseClickEventFunc(ev)
    if self.button == self.rbButton:  
      ev.keymod = KeyboardModifiers()
      if ev.keymod == self.xyKeyMod[0]:
        self.unZoom(0)
      elif ev.keymod == self.xyKeyMod[1]:
        self.unZoom(1)
      elif ev.keymod == self.xyKeyMod[2]:
        self.unZoom(2)
  def mouseDoubleClickEvent(self, ev):
    if not(self.GUI):
      if self.GUI is not None:
        BasePlotClass.mouseDoubleClickEvent(self, ev)
      return  
    if self.mouseDoubleClickEventFunc is not None:
      ev.X, ev.Y = self.mapxy(ev)
      self.sender = self
      self.mouseDoubleClickEventFunc(ev)
    BasePlotClass.mouseDoubleClickEvent(self, ev)
  def keyPressEvent(self, ev = None):
    if not(self.GUI):
      if self.GUI is not None:
        if self.Cursor is not None : # remove cursors if escaping
          if ev is not None:
            if ev.key() == self.escKey:
              self.setCursor(ev)
              return
        BasePlotClass.keyPressEvent(self, ev)
      return  
    if self.keyPressEventFunc is not None:
      ev.sender = self
      self.keyPressEventFunc(ev)
    if self.moved and self.button == self.rbButton: # we're mid-event
      ev.keymod = KeyboardModifiers()
      if ev.keymod in self.xyKeyMod or int(ev.keymod) == int(self.xyKeyMod[2]):
        return self.mouseDragEvent(ev)
  def keyReleaseEvent(self, ev):
    if not(self.GUI):
      if self.GUI is not None:
        BasePlotClass.keyReleaseEvent(self, ev)
      return  
    if self.moved and self.button == self.rbButton: # we're mid-event
      ev.keymod = 0
      self.setRB(ev)
    BasePlotClass.keyReleaseEvent(self, ev)
  def viewRangeChanged(self, view, _xxyy):
    if not(self.GUI):
      if self.GUI is not None:
        BasePlotClass.viewRangeChanged(self)
      return  
    if self.lockxy: return # allow over-riding automatic calls
    self.firstEvent(None)
    BasePlotClass.viewRangeChanged(self)
    self.xxyy = self.bound(_xxyy)
    if self.boundx:
      self.setXRange(self.xxyy[0][0], self.xxyy[0][1], 0., True) 
    if self.boundy:
      self.setYRange(self.xxyy[1][0], self.xxyy[1][1], 0., True)
    if self.viewRangeChangedFunc is not None:
      self.viewRangeChangedFunc(_xxyy)

class tabl (BaseTablClass): # adds mouse clickability to Base Tabl Class
  moved = False
  def __init__(self, *args, **_kwds):
    self.form = None # QMainWindow
    self.lbox = None # LayoutWidget
    self.view = None # ViewBox
    add2lbox = False
    kwds = dict(_kwds)
    add2lbox = True
    if 'parent' in kwds:
      self.lbox = kwds['parent']
      self.form = self.lbox.parent()
      del kwds['parent']
    else:
      self.form = BaseFormClass()
      self.lbox = lbox()
      self.form.setCentralWidget(self.lbox)
    BaseTablClass.__init__(self, *args, **kwds)  
    if add2lbox: 
      self.lbox.add(self)
      self.lbox.setKeyReleaseEventFunc(self.keyReleaseEvent)
    self.setMouseClickEventFunc()
  def show(self):
    if self.form is not None: return self.form.show()
    if self.lbox is not None: return self.lbox.show()
    if self.view is not None: return self.view.show()
    return self.show()
  def setMouseClickEventFunc(self, _mouseClickEventFunc = None):
    self.mouseClickEventFunc = _mouseClickEventFunc
  def mouseReleaseEvent(self, ev): 
    ev.row = self.currentRow()
    ev.col = self.currentColumn()
    if self.mouseClickEventFunc is not None:
      self.mouseClickEventFunc(ev)
    BaseTablClass.mouseReleaseEvent(self, ev)

class imag (BaseImagClass): # adds mouse clickability to Base Imag Class
  moved = False
  def __init__(self, *args, **_kwds):
    self.form = None # QMainWindow
    self.lbox = None # LayoutWidget
    self.view = None # ViewBox
    add2lbox = False
    kwds = dict(_kwds)
    add2lbox = True
    if 'parent' in kwds:
      self.lbox = kwds['parent']
      self.form = self.lbox.parent()
      del kwds['parent']
    else:
      self.form = BaseFormClass()
      self.lbox = lbox()
      self.form.setCentralWidget(self.lbox)
    BaseImagClass.__init__(self, *args, **kwds)  
    if add2lbox: 
      self.lbox.add(self)
      self.lbox.setKeyReleaseEventFunc(self.keyReleaseEvent)
    self.setMouseClickEventFunc()
  def show(self):
    if self.form is not None: return self.form.show()
    if self.lbox is not None: return self.lbox.show()
    if self.view is not None: return self.view.show()
    return self.show()
  def setMouseClickEventFunc(self, _mouseClickEventFunc = None):
    self.mouseClickEventFunc = _mouseClickEventFunc
  def mousePressEvent(self, ev): 
    if self.mouseClickEventFunc is not None:
      self.mouseClickEventFunc(ev)
    BaseImagClass.mousePressEvent(self, ev)

class surf (BaseSurfClass):  # interim solution until we have vispy
  xyzs = [3., 3., 3.]
  def __init__(self, *args, **kwds):
    BaseSurfClass.__init__(self, *args, **kwds)
  def setData(self, x=None, y=None, z=None, colors=None):
    mino = 1e-300
    if x is not None:
      xmn, xmx = x.min(), x.max()
      x = (2.*(x-0.5*(xmn+xmx)))/(xmx-xmn+mino)
      x *= self.xyzs[0]
    if y is not None:
      ymn, ymx = y.min(), y.max()
      y = (2.*(y-0.5*(ymn+ymx)))/(ymx-ymn+mino)
      y *= self.xyzs[1]
    if z is not None:
      z /= z.max()+mino
      z *= self.xyzs[2]
    BaseSurfClass.setData(self, x, y, z, colors)

