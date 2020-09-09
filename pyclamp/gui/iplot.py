# isubplot

import os
import sys
import matplotlib as mpl
#'''
if mpl.get_backend() != 'Qt4Agg':
  mpl.use('Qt4Agg')
  import PyQt4.QtCore as QtCore
#'''
import matplotlib.pyplot as mp
import matplotlib.widgets as mw
from matplotlib.patches import Rectangle
import numpy as np
from lsfunc import *
from time import time
from mpl_toolkits.mplot3d import Axes3D
from fpfunc import *

DISPLAY = "DISPLAY"

try:
  DISPLAY = os.environ[DISPLAY]
except KeyError:
  DISPLAY = None

LEFTBUTTON = 1
MIDDLEBUTTON = 2
RIGHTBUTTON = 3
IFIGURELIST = []
CONSOLECOL = {'w':97, 'm':95, 'r':91, 'y':93, 'g':92, 'c':96, 'b':94, 'k':30, 'e':1,
              'W':97, 'M':95, 'R':91, 'Y':93, 'G':92, 'C':96, 'B':94, 'K':30, 'E':1}

def ttysize(spec = 'stty size'):
  rc = os.popen(spec, 'r').read().split()
  try:
    RC = int(rc[0]), int(rc[1])
  except IndexError:
    RC = None, None
  return RC

def colstr(_str, col = None):
  if col is None: return _str
  strcol = '\033[%sm'%CONSOLECOL[col] if type(col) is str else '\033[%sm'%col
  return strcol + _str + '\033[%sm'%0

def figexists(_fI):
    try:
        return mp.fignum_exists(_fI.number)
    except AttributeError:
        return False  

def scf(_fI):
    mp.figure(_fI.number)

def figclear(_fI = None):
  fI = _fI if _fI is not None else mpl.gcf()
  ax = fI.axes
  for i in range(len(ax)):
    fI.delaxes(ax[i])

def showticks(ax = None, xon = True, yon = True):
  if ax is None: ax = mp.gca()
  on = xon
  for tick in ax.xaxis.get_major_ticks():
    tick.tick1On = on
    tick.tick2On = on
    tick.label1On = on
    tick.label2On = on
  on = yon  
  for tick in ax.yaxis.get_major_ticks():
    tick.tick1On = on
    tick.tick2On = on   
    tick.label1On = on
    tick.label2On = on    
    
def showspines(spec = 'NESW'):
  if type(spec) is str: spec = spec.upper()
  ax = mp.gca()
  N = listfind(spec, 'N') >= 0
  E = listfind(spec, 'E') >= 0
  S = listfind(spec, 'S') >= 0
  W = listfind(spec, 'W') >= 0
  if not(N): ax.spines['top'].set_visible(False)
  if not(E): ax.spines['right'].set_visible(False)
  if not(S): ax.spines['bottom'].set_visible(False)
  if not(W): ax.spines['left'].set_visible(False)
  if N: ax.get_xaxis().tick_top()  
  if E: ax.get_yaxis().tick_right()
  if S: ax.get_xaxis().tick_bottom()  
  if W: ax.get_yaxis().tick_left()

def dirnticks(ax = None, xdr = 'out', ydr = 'out'):
  if ax is None: ax = mp.gca()
  dr = xdr
  for tick in ax.xaxis.get_major_ticks():
    tick._apply_params(tickdir = dr)
  for tick in ax.xaxis.get_minor_ticks():
    tick._apply_params(tickdir = dr)
  dr = ydr  
  for tick in ax.yaxis.get_major_ticks():
    tick._apply_params(tickdir = dr)
  for tick in ax.yaxis.get_minor_ticks():
    tick._apply_params(tickdir = dr)

def label(x, y, txt, pos = None, **kwArgs):
  if type(pos) is str: pos = pos.upper()
  ha, va = 'center', 'center'
  if pos == 'N':
    ha, va = 'center', 'bottom'
  elif pos == 'NE': 
    ha, va = 'left', 'bottom'
  elif pos == 'E':
    ha, va = 'left', 'center'
  elif pos == 'SE':
    ha, va = 'left', 'top'
  elif pos == 'S':
    ha, va = 'center', 'top'
  elif pos == 'SW':
    ha, va = 'right', 'top'
  elif pos == 'W':
    ha, va = 'right', 'center'
  elif pos == 'NW':
    ha, va = 'right', 'bottom'
  return mp.text(x, y, txt, horizontalalignment = ha, verticalalignment = va, **kwArgs)   

def cornertitle(txt, pos = 'NW', drxy = [0.05, 0.05], **kwArgs):
  pos = pos.upper()
  ax = mp.gca().axis()
  Dx = ax[1] - ax[0]
  Dy = ax[3] - ax[2]
  if pos == 'NE':
    x = ax[0] + (1. + drxy[0]) * Dx
    y = ax[2] + (1. + drxy[1]) * Dy
  elif pos == 'SE':
    x = ax[0] + (1. + drxy[0]) * Dx
    y = ax[2] - drxy[1] * Dy
  elif pos == 'SW':
    x = ax[0] - drxy[0] * Dx
    y = ax[2] - drxy[1] * Dy 
  else:
    x = ax[0] - drxy[0] * Dx
    y = ax[2] + (1. + drxy[1]) * Dy
  return label(x, y, txt, None, **kwArgs)

def plotsurf(ax, x, y, Z, *args, **kwargs):
  if ax is None: ax = mp.gca()
  nx = len(x);
  ny = len(y);
  X = np.tile(x.reshape(nx, 1), (1, ny))
  Y = np.tile(y.reshape(1, ny), (nx, 1))
  return ax.plot_surface(X, Y, Z, *args, **kwargs)
    
def xplot(x0, y0, dx, dy, *args):
  ph = mp.plot([x0-dx, x0+dx], [y0, y0], *args)
  pv = mp.plot([x0, x0], [y0-dy, y0+dy], *args)
  return ph, pv

def exes(position = 'b', n = 0, N = 1, border = 0.1, clearance = 0.04, extent = 0.04): # returns an axes around a figure's border
  ax = np.zeros(4, dtype = float)
  edge = border - clearance
  d = 1.0 - (border*2.0)
  dN = d / float(N)
  dnN = dN * float(n)
  if position == 'b' or position == 't':
    ax[0] = border + dnN
    ax[2] = dN
    ax[3] = extent
    if position == 'b':
      ax[1] = edge - extent
    else:
      ax[1] = 1 - edge
  elif position == 'l' or position == 'r':
    ax[1] = border + dnN    
    ax[2] = extent
    ax[3] = dN
    if position == 'l':
      ax[0] = edge - extent
    else:
      ax[0] = 1 - edge
  else:
    print("Warning from exes: position variable not set correctly - no axes returned")
    return None
  return mp.axes(ax)

def rgbJet(x):
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
    return np.array( (r, g, b) )
  
  
def setAxesColours(axInst = None, fgColour = None, bgColour = None):
  if (axInst == None): axInst = mp.gca()
  if fgColour != None:
    axInst.tick_params('both', color = fgColour, labelcolor = fgColour)
    for child in axInst.get_children(): 
      if isinstance(child, mpl.spines.Spine): 
        child.set_color( fgColour )   
    axInst.title.set_color( fgColour )
    axInst.xaxis.label.set_color( fgColour )
    axInst.yaxis.label.set_color( fgColour )
  if bgColour != None:
    axInst.set_axis_bgcolor(bgColour)   
  
def ret_axes_width(figInst, axInst):
  if (figInst != None and axInst != None):
    return int(axInst.get_position().width * figInst.get_dpi() * figInst.get_size_inches()[0])
  else:
    return int(0)  

def ret_axes_height(figInst, axInst):
  if (figInst != None and axInst != None):
    return int(axInst.get_position().height * figInst.get_dpi() * figInst.get_size_inches()[1])
  else:
    return int(0)  
            
class figs:
  defnmax = 25
  def __init__(self, _nmax = None):
    self.setnmax(_nmax)
  def setnmax(self, _nmax = None):
    if _nmax is None: _nmax = self.defnmax
    self.nmax = int(_nmax)
    self.nrow = int(np.floor(np.sqrt(float(self.nmax))))
    self.ncol = int(np.ceil(float(self.nmax)/float(self.nrow)))
    self.i = 0
    self.f = []
  def newPlot(self):
    if self.i == 0:
      self.f.append(mp.figure())
    self.i += 1
    ax = self.f[-1].add_subplot(self.nrow, self.ncol, self.i)
    if self.i == self.nmax:
      self.i = 0
    return ax

class GUIEllipse:
  def_n = 128
  def __init__(self, _xyabps = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], _r = 1.0, _n = None):
    self.initialise(_xyabps, _r, _n)
  def initialise(self, _xyabps = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0], _r = 1.0, _n = None):
    self.set_xyabps(_xyabps)
    self.set_r(_r)
    self.set_n(_n)
  def set_xyabps(self, _xyabps = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]):
    self.xyabps = _xyabps
  def set_r(self, _r = 1.0):
    self.r = _r
  def set_n(self, _n = None):
    if _n is None: _n = self.def_n
    self.n = _n
    self.t = np.linspace(-np.pi, np.pi, self.n)
  def calcEllipse(self, _t = None):
    if _t is None: _t = self.t
    return normel2cart(_t, self.r, self.xyabps)
  def calcNormel(self, x, y):
    return cart2normel(x, y, self.xyabps)    
          
class pbfig:
  mino = 1e-300
  mindt = 0.03 # secs 
  tmin = 0.0
  deffigsize = [8.125, 6.125]
  defbarh = 0.2 # relative to figure
  N = 0
  non = None
  ft = None
  fI = None
  reprint = True
  useconsole = False
  rw = 1
  defcw = 50 # default column (`virtual') width if tty unavailable
  ldd = 0 # last denominator decremented
  st = None
  cwpt = 0.3 # polling interval to check console column width (secs)
  def __init__(self, _figtitle = None, _tit = None, _den = 100, _col = 'b', _useconsole = False):
    if _useconsole: # easiest to do at this stage
      self.useConsole(_useconsole)
    self.isint = mp.isinteractive()    
    self.figsize = [self.deffigsize[0], self.deffigsize[1]*self.defbarh]
    if not(_figtitle is None): 
      self.setTitle(_figtitle)      
    else:
      self.figtitle = None
    if _tit is not None:
      self.setup(_tit, _den, _col)
  def useConsole(self, _useconsole = True):
    self.useconsole = _useconsole
    if self.useconsole and self.fI is not None: # remove any created figure
      mp.close(self.fI)
      self.N = 0
      self.setSubPlots()
      self.forceupdate() # ensure first update
  def setup(self, _tit = 'Please wait...', _den = 100, _col = 'b'): 
    if not(type(_tit) is list): _tit = [_tit];
    if not(type(_den) is list): _den = [_den];
    #if not(type(_col) is list): _col = [_col];
    self.n = max(len(_tit), len(_den), len(_col))
    self.I = range(self.n)
    self.non = [None] * self.n
    self.tit = repl(_tit, self.n)
    self.den = repl(_den, self.n)
    self.col = repl(_col, self.n)
    self.dnd = np.array(self.den, dtype = int) - 1 # denomator decremented
    self.num = [0.0] * self.n
    self.pct = [-1] * self.n
    self.lti = np.empty(self.n, dtype = int)
    for i in self.I: 
      self.lti[i] = len(self.tit[i])
      self.den[i] = 0.01 * float(self.den[i]) 
    self.setSubPlots()
    self.forceupdate() # ensure first update
  def setlast(self, _titlast = 'Please wait', _denlast = 100, _collast = 'b'):
    self.tit[-1] = _titlast
    self.den[-1] = 0.01 * float(_denlast)
    self.dnd[-1] = _denlast - 1
    self.col[-1] = _collast
    self.num[-1] = 0.0
    self.pct[-1] = -1
    if not(self.useconsole):
      mp.sca(self.aI[-1])
      mp.title(self.tit[-1])
    self.updatelast(0)
  def forceupdate(self, _num = None):
    self.tmin = time() - self.mindt
    if not(_num is None): self.update(_num)
  def update(self, _num):
    t = time()
    if not(type(_num) is list): _num = [_num]
    redraw = _num[-1] == self.dnd[-1] 
    if t < self.tmin and not(redraw): return
    if redraw:
      self.pct[-1] = 100
      self.updateSubPlot(self.n-1)
      self.tmin = t - self.mindt # force a next update
      self.df()
      return
    self.tmin = t + self.mindt
    for i in self.I:
      if _num[i] is not None:
        _pct = round(float(_num[i]) / (self.mino + self.den[i]))
        if self.pct[i] != _pct:
          redraw = True
          self.pct[i] = int(_pct)
          self.updateSubPlot(i)
          if i < self.n - 1:
            self.reprint = 1
    if redraw: 
      self.df()
      if not(self.useConsole):
        if mpl.get_backend() == "Qt4Agg":
          QtCore.QCoreApplication.processEvents()
  def updatelast(self, _numlast):
    self.non[-1] = _numlast
    self.update(self.non)
  def close(self):
    if not(self.useconsole):
      mp.close(self.fI)     
    else:
      print('\n')
    if not(self.isint): mp.ioff()
    self.N = 0 
  def setTitle(self, _figtitle = None):
    self.figtitle = _figtitle
    if _figtitle is not None: 
      self.ft = colstr(self.figtitle, 'e')
    if self.figtitle is None: return
    if self.fI is None: return
    if type(self.figtitle) is not(str): return
    if not(self.useconsole):
      self.fI.canvas.set_window_title(self.figtitle)
      self.df()
      return
  def setSubPlots(self):
    if (self.N != self.n):
      self.newFigure(self.figtitle)
      self.addSubPlots()
    self.updateTitles()  
    self.update([0] * self.n) 
  def newFigure(self, _figtitle = None):
    if _figtitle is None: _figtitle = self.figtitle 
    newsize = [self.figsize[0], self.figsize[1]*float(self.n)]
    self.fI = None
    if not(self.useconsole):
      try:
        mp.ion()
        with mpl.rc_context({'toolbar':False}):
          self.fI = mp.figure(figsize = newsize)    
          self.fI.canvas.draw()
        if mpl.get_backend() == "Qt4Agg":
          QtCore.QCoreApplication.processEvents()
      except Exception, e: # catches all exceptions
        self.fI = None
        self.useConsole()
    if self.useconsole:
      self.fI = None
      self.df = self.drawConsole
    else:
      self.fI.clf()        
      try:
        self.fI.set_size_inches(newsize, forward = True)        
      except Exception, e: # catches all exceptions
        pass
      self.df = self.fI.canvas.draw
    self.setTitle(_figtitle)
  def updateTitles(self):  
    if not(self.useconsole):
      for i in self.I:
        mp.sca(self.aI[i])
        mp.title(self.tit[i])
      return  
    # Console specific code 
    maxlti = self.lti.max()
    if self.rw is not None: # prevent it logging errors _every_ time outside tty
      self.rw, self.cw = ttysize()
    if self.cw is None: self.cw = self.defcw
    self.cwt = time()
    '''
    TEMPLATE:
    EMBOLDENED TITLE
    SUBTITLE(variable space)__ or _.[4 characters for: space space 0% or space space XX% or done]
    '''
    self.pbwid = 100
    maxpbwid = self.cw - maxlti - 4
    if maxpbwid < 50:
      self.pbwid = 0
    elif maxpbwid < 100:
      self.pbwid = 50
    elif maxpbwid < 150:
      self.pbwid = 100
    elif maxpbwid < 200:
      self.pbwid = 150
    elif maxpbwid < 250:
      self.pbwid = 200
    elif maxpbwid < 300:
      self.pbwid = 250
    else:
      self.pbwid = 300
    self.vs = np.maximum(0, self.cw - (self.lti + self.pbwid + 4))
    self.st = [''] * self.n
  def addSubPlots(self):
    self.N = self.n  
    if not(self.useconsole):
      self.aI = [[]] * self.n
      self.pI = [[]] * self.n
      self.yI = [[]] * self.n
      for i in self.I:
        self.aI[i] = self.fI.add_subplot(self.n*2, 1, 2*(i+1)) 
        showticks(self.aI[i], False, False)
        self.aI[i].set_xlim(0, 100)
        self.aI[i].set_ylim(0, 1)
        self.pI[i] = Rectangle([0, 0], 0, 1, facecolor = self.col[i], fill = True)
        self.aI[i].add_patch(self.pI[i])
        self.aI[i].yaxis.set_label_position("right")
        self.yI[i] = mp.ylabel('      0%')
        self.yI[i].set_rotation(0)     
      if mpl.get_backend() == "Qt4Agg":
        QtCore.QCoreApplication.processEvents()
      return    
  def updateSubPlot(self, i):      
    if not(self.useconsole):
      self.pI[i].set_width(self.pct[i]) 
      self.yI[i].set_text('      ' + str(self.pct[i]) + '%')
      if mpl.get_backend() == "Qt4Agg":
        QtCore.QCoreApplication.processEvents()
      return
    # Console specific code
    t = time() 
    if t - self.cwt >= self.cwpt:
      self.cwt = t
      if self.rw is not None: 
        _, _cw = ttysize()
        if self.cw != _cw: self.updateTitles()
    sti0 = self.tit[i] + ' ' * self.vs[i]
    if DISPLAY:
      __ = '~'
      _ = '-'
    else:
      __ = '%'
      _ = '/'
    if self.pbwid == 0:
      sti1 = ''
    elif self.pbwid == 50:
      sti1 = __ * int(self.pct[i] / 2) + _ * (self.pct[i] % 2) 
    elif self.pbwid == 100:
      sti1 = __ * self.pct[i]
    elif self.pbwid == 150:
      pcti2 = self.pct[i] * 3
      sti1 = __ * int(pcti2 / 2) + _ * (pcti2 % 2) 
    elif self.pbwid == 200:
      sti1 = (__ + __) * self.pct[i]
    elif self.pbwid == 250:
      pcti2 = self.pct[i] * 5
      sti1 = __ * int(pcti2 / 2) + _ * (pcti2 % 2) 
    elif self.pbwid == 300:
      sti1 = (__ + __ + __) * self.pct[i]
    sti1 += ' ' * max(0, self.pbwid - len(sti1)) 
    sti2 = str(self.pct[i]) + '%' if self.pct[i] < 100 else 'done'
    sti2 = ' ' * (4 - len(sti2)) + sti2
    if self.rw is not None:
      self.st[i] = sti0 + colstr(sti1, self.col[i]) + sti2
    else:
      self.st[i] = sti0 + sti1 + sti2
  def drawConsole(self):  
    if self.reprint:
      if self.rw is None:
        sys.stdout.write('\n')
        if self.figtitle is not None:
          sys.stdout.write(self.figtitle)
          sys.stdout.write('\n')
        for i in self.I[:-1]:
          sys.stdout.write(self.st[i])
      else:
        if self.ft is not None: 
          print('\n')
          print(self.ft)
        else:
          print('\n',)
        for i in self.I[:-1]:
          print(self.st[i] )
      if self.rw is not None:   
        sys.stdout.write(self.st[-1] + '\r')
      else:  
        sys.stdout.write(self.st[-1])
      self.reprint = False
      return
    sys.stdout.flush()
    if self.rw is not None:
      sys.stdout.write(self.st[-1] + '\r')
    else:  
      sys.stdout.write(self.st[-1] + '\n')

class iaxes:
  def __init__(self, _aI = None):
    self.initialise(_aI)
  def initialise(self, _aI = None):  
    self.aI = _aI
  def refreshAxes(self):  
    if self.aI == None: return
    self.xxyy = self.aI.axis()
    XXYY = self.aI.bbox.extents
    self.XXYY = [XXYY[0], XXYY[2], XXYY[1], XXYY[3]]
    self.calcConversion()  
  def calcConversion(self):
    mino = 1e-300
    if self.aI == None: return
    x0 = float(self.xxyy[0])
    dx = float(self.xxyy[1] - self.xxyy[0])
    y0 = float(self.xxyy[2])
    dy = float(self.xxyy[3] - self.xxyy[2])
    X0 = float(self.XXYY[0])
    DX = float(self.XXYY[1] - self.XXYY[0])
    Y0 = float(self.XXYY[2])
    DY = float(self.XXYY[3] - self.XXYY[2])
    X2x = dx / (mino + DX)
    self.X2x = [X2x, x0 - X0*X2x] 
    x2X = DX / (mino + dx)
    self.x2X = [x2X, X0 - x0*x2X] 
    Y2y = dy / (mino + DY)
    self.Y2y = [Y2y, y0 - Y0*Y2y] 
    y2Y = DY / (mino + dy)
    self.y2Y = [y2Y, Y0 - y0*y2Y]
  def xy2XY(self, x = None, y = None):
    X = x
    Y = y
    if self.aI is None: return X, Y
    if not(X is None):
      X = float(X) * self.x2X[0] + self.x2X[1]
    if not(Y is None):
      Y = float(Y) * self.y2Y[0] + self.y2Y[1]
    return X, Y
  def XY2xy(self, X = None, Y = None):
    x = X
    y = Y
    if self.aI is None: return x, y
    if not(X is None):
      x = float(x) * self.X2x[0] + self.X2x[1]
    if not(Y is None):
      y = float(y) * self.Y2y[0] + self.Y2y[1]
    return x, y  
  def boundxy(self, x = None, y = None):
    if x < self.xxyy[0]:
      x = self.xxyy[0]
    elif x > self.xxyy[1]:
      x = self.xxyy[1]
    if y < self.xxyy[2]:
      y = self.xxyy[2]
    elif y > self.xxyy[3]:
      y = self.xxyy[3]  
    return x, y  
  def boundEvent(self, event):
    if self.aI == event.inaxes: return    
    event.xdata, event.ydata = self.XY2xy(event.x, event.y)
    event.xdata, event.ydata = self.boundxy(event.xdata, event.ydata)
    
class ifigure:    
  fI = None  
  canv = None  
  cidde = None
  cidre = None
  cidbpe = None
  cidbre = None
  cidmne = None  
  cidkpe = None
  cidse = None
  cidpe = None  
  button_pressed = False
  motion_notified = False
  guiisp = None
  def __init__(self, _fI = None):
    self.initialise(_fI)
  def initialise(self, _fI = None):  
    self.disconnectISubPlots()
    self.setFigure(_fI)
  def setFigure(self, _fI = None):
    if _fI is None: return
    self.fI = _fI
    for ifigure in IFIGURELIST:
      if self.fI == ifigure[1]:
        raise ValueError("Attempting multiple ifigures instances for same figure")
    self.canv = self.fI.canvas
    self.connectCID()
  def connectCID(self, cid = None, evnt = None, func = None):
    self.cidde = self.canv.mpl_connect('draw_event', self.on_draw_event)
    self.cidre = self.canv.mpl_connect('resize_event', self.on_resize_event)
    self.cidbpe = self.canv.mpl_connect('button_press_event', self.on_button_press_event)
    self.cidbre = self.canv.mpl_connect('button_release_event', self.on_button_release_event)
    self.cidmne = self.canv.mpl_connect('motion_notify_event', self.on_motion_notify_event)
    self.cidkpe = self.canv.mpl_connect('key_press_event', self.on_key_press_event)  
    self.cidse = self.canv.mpl_connect('scroll_event', self.on_scroll_event)
    self.cidpe = self.canv.mpl_connect('pick_event', self.on_pick_event)
  def disconnectCID(self, cid = None):
    if self.cidde is not None: self.canv.mpl_disconnect(self.cidde); self.cidde = None
    if self.cidre is not None: self.canv.mpl_disconnect(self.cidre); self.cidre = None
    if self.cidbpe is not None: self.canv.mpl_disconnect(self.cidbpe); self.cidbpe = None
    if self.cidbre is not None: self.canv.mpl_disconnect(self.cidbre); self.cidbre = None
    if self.cidmne is not None: self.canv.mpl_disconnect(self.cidmne); self.cidmne = None
    if self.cidkpe is not None: self.canv.mpl_disconnect(self.cidkpe); self.cidkpe = None
    if self.cidse is not None: self.canv.mpl_disconnect(self.cidse); self.cidse = None
    if self.cidpe is not None: self.canv.mpl_disconnect(self.cidpe); self.cidpe = None    
  def disconnectISubPlots(self):
    self.Ai = []  
    self.isp = []
    self.nsp = 0    
  def disconnectISubPlot(self, _isp = None):
    if _isp is None: return
    ID = listfind(self.isp, _isp)
    if ID is not None:
      del self.Ai[ID];
      del self.isp[ID];
      self.nsp = len(self.isp)
  def connectISubPlot(self, _isp = None):
    if _isp is None: return
    for sp in self.isp: 
      if sp == _isp:
        raise ValueError("isubplot instance already added to ifigure class.")  
    for ai in self.Ai: 
      if ai == _isp.aI:
        raise ValueError("AxesSubplot instance is already added to figure class")
    self.Ai.append(_isp.aI)
    self.isp.append(_isp)    
    self.nsp = len(self.isp)
  def eventID(self, event):
    return listfind(self.Ai, event.inaxes)
  def on_draw_event(self, event):
    try:
      for _isp in self.isp:
        _isp.on_draw_event(event)
    except ValueError:
      self.disconnectISubPlot(_isp)
      self.on_draw_event(event)
  def on_resize_event(self, event):
    try:
      for _isp in self.isp:
        _isp.on_resize_event(event)
    except ValueError:
      self.disconnectISubPlot(_isp)
      self.on_resize_event(event)
  def on_button_press_event(self, event):
    ID = self.eventID(event)
    if ID is not None:
      self.guiisp = self.isp[ID]
      self.button_pressed = True
      self.motion_notified = False
      self.guiisp.on_button_press_event(event)
  def on_button_release_event(self, event):
    self.button_pressed = False
    if self.guiisp is not None:
      self.guiisp.on_button_release_event(event)
  def on_motion_notify_event(self, event):
    self.motion_notified = True
    if self.button_pressed and self.guiisp is not None:
      self.guiisp.on_motion_notify_event(event)
    else: 
      ID = self.eventID(event)
      if ID is not None:
        self.guiisp = self.isp[ID] # store relevant subplot to append to pickevent
        self.isp[ID].on_motion_notify_event(event)
      else:
        self.guiisp = None # if left all relevant axes  
  def on_key_press_event(self, event):
    for _isp in self.isp:
      _isp.on_key_press_event(event)
  def on_scroll_event(self, event):
    ID = self.eventID(event)
    if ID is not None:
      self.isp[ID].on_scroll_event(event)      
  def on_pick_event(self, event):
    event.inaxes = self.guiisp.aI
    ID = self.eventID(event)    
    if ID is not None:
      self.isp[ID].on_pick_event(event)      
      
class isubplot:  
  aI = None
  fI = None
  iF = None
  defrbFormat = ""
  defswSens = 2.0 # in units of percentage for y-panning
  defswOpts = 1 # 0 = neither, 1 = zooms, 2 = pans Y  
  def __init__(self, argIn0 = None, argIn1 = None, argIn2 = None):
    self.initialise()
    self.subplot(argIn0, argIn1, argIn2)
  def initialise(self):
    if self.aI != None: self.aI = None 
    self.isint = mp.isinteractive()    
    self.setRbFormat(self.defrbFormat)
    self.setSW(self.defswSens, self.defswOpts)
    self.initCursor()    
    self.initGUIPlots()      
    self.initGUI()   
  def setRbFormat(self, _rbFormat = None):
    if _rbFormat is not None: self.rbFormat = _rbFormat
    self.rbPlotted = 0  
  def setSW(self, _swSens = None, _swOpts = None):
    if _swSens is not None: self.swSens = _swSens
    if _swOpts is not None: self.swOpts = _swOpts   
  def initCursor(self):
    self.cursorHV = [False, False]
    self.cursor = None
    self.ncursor = 0
    self.curscounter = 0 
    self.cursorCBFunc = None
    self.cursorPairHV = None
    self.cursorPairCBFunc = None     
  def initGUIPlots(self):
    self.stillClickFunc = []  # Agreed - not strictly related to plots  
    self.plots = []
    self.plotsx = []
    self.plotsy = []
    self.plotsn = 0  
    self.plotaxwh = None
    self.plotsInitialised = []
    self.plotsInitFunc = []  
    self.plotsCalcFunc = []
    self.plotsEvntList = []
    self.plotsFuncList = []
    self.plotsPlotStrg = [] 
    self.plotCID = []
    self.plotBPEFuncList = [] 
    self.plotBREFuncList = []
    self.plotMNEFuncList = []
    self.plotKPEFuncList = []
    self.plotPEFuncList = []
    self.plotGUIKey = {'button_press_event': self.plotBPEFuncList, 
                       'button_release_event': self.plotBREFuncList,  
                       'motion_notify_event': self.plotMNEFuncList,
                       'key_press_event': self.plotKPEFuncList,
                       'pick_event': self.plotPEFuncList}
  def initGUI(self):
    self.xbeg = None
    self.ybeg = None
    self.xend = None
    self.yend = None
    self.xran = None
    self.yran = None        
    self.xmid = None
    self.ymid = None
    self.xwid = None
    self.ywid = None
    self.xyGUI = [True, True, True, True, 1]
    self.GUIButton = 0
    self.GUIBypass = False
    self.GUICursor = False
    self.origxlim = [-np.inf, np.inf]
    self.origylim = [-np.inf, np.inf]   
    self.origView = True
    self.disconnectGUI()
  def connectGUI(self, _fI):
    self.fI = _fI
    if self.iF is None:
      ID = None      
      for i in range(len(IFIGURELIST)):
        ifigurelist = IFIGURELIST[i]
        if type(ifigurelist) is list:
          if len(ifigurelist) == 2:
            if self.fI == ifigurelist[1]:
              ID = i
      if ID is None:
        IFIGURELIST.append([ifigure(self.fI), self.fI])
        ID = len(IFIGURELIST) - 1
      self.iF = IFIGURELIST[ID][0]
    self.iF.connectISubPlot(self)  
  def disconnectGUI(self):
    self.bgbbox = None
    self.bgGUIbbox = None
    if self.iF is None: return
    self.iF.disconnectISubPlot(self)
  def reconnectGUI(self, _fI = None):
    if _fI is None: _fI = self.fI
    self.initGUI()
    self.connectGUI(_fI)       
  def subplot(self, argIn0 = None, argIn1 = None, argIn2 = None):
    self.aI = None
    if argIn2 != None:
      self.aI = mp.subplot(argIn0, argIn1, argIn2)
    elif argIn0 != None:
      self.aI = mp.subplot(argIn0)
    elif argIn1 == None:
      self.aI = mp.gca()
    if self.aI is not None:
      self.iaI = iaxes(self.aI)
      self.connectGUI(self.aI.figure)
    return self.aI 
  def ret_axes(self):
    return self.aI  
  def setGUI(self, canZoomX = None, canZoomY = None, canPanX = None, canPanY = None, swOpts = None):
    if canZoomX is not None: self.xyGUI[0] = canZoomX
    if canZoomY is not None: self.xyGUI[1] = canZoomY
    if canPanX is not None: self.xyGUI[2] = canPanX
    if canPanY is not None: self.xyGUI[3] = canPanY
    if swOpts is not None:
      self.xyGUI[4] = swOpts
    else:
      if self.xyGUI[0] or self.xyGUI[1]:
        self.xyGUI[4] = 1
      elif self.xyGUI[2] or self.xyGUI[3]:
        self.xyGUI[4] = 2      
  def addStillClickFunc(self, _stillClickFunc):
    self.stillClickFunc.append(_stillClickFunc)
  def setCursor(self, _cursorHV = [False, False], _cursorCBFunc = None, **kwargs):
    if not(type(_cursorHV) is list): _cursorHV = [_cursorHV]*2
    _GUICursor = _cursorHV[0] or _cursorHV[1]
    if self.cursor != None:
      self.cursor.clear(None)
      self.cursor = None
    self.cursorHV = _cursorHV
    self.GUICursor = _GUICursor 
    if not(_GUICursor): 
      self.redraw(False, True)    
      return # this prevents over-writing self.cursorCBFunc
    self.cursorCBFunc = _cursorCBFunc
    if self.cursorCBFunc != self.cursorPair or not(self.curscounter):
      self.cursor = mw.Cursor(self.aI, useblit=True, **kwargs)
      self.cursor.horizOn = self.cursorHV[0]
      self.cursor.vertOn = self.cursorHV[1]
  def cursorPair(self, _cursorHV = [True, True], _cursorPairCBFunc = None, **kwargs):
    if not(type(_cursorHV) is list):
      if type(_cursorHV) is bool or type(_cursorHV) is int:
        _cursorHV = [_cursorHV] * 2  
    if type(_cursorHV) is list:
      if not( type(_cursorHV) is list): _cursorHV = [_cursorHV] * 2
      self.XYGUI = []
      self.curscounter = 0
      self.ncursor = 2
      self.cursorPairHV = _cursorHV
      self.cursorPairCBFunc = _cursorPairCBFunc
      self.kwArgs = kwargs
      self.setCursor(self.cursorPairHV, self.cursorPair, **self.kwArgs)
    else:
      self.curscounter += 1
      self.XYGUI.append([_cursorHV.xdata, _cursorHV.ydata])    
      if self.curscounter == self.ncursor: 
        self.GUICursor = 0 # to instruct plotRB to delete
        self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend) 
        self.curscounter = 0
        self.ncursor = 0
        _cursorPairCBFunc = self.cursorPairCBFunc
        _cursorPairCBFunc(_cursorHV)
        self.cursorPairCBFunc = None
      else:
        self.setCursor(self.cursorPairHV, self.cursorPair, **self.kwArgs)
  def plotRB(self, x0 = None, y0 = None, x1 = None, y1 = None):
    if not(self.GUICursor): 
      if not(self.xyGUI[0]):
        x0 = self.xlim[0]
        x1 = self.xlim[1]
      if not(self.xyGUI[1]):
        y0 = self.ylim[0]
        y1 = self.ylim[1]  
    else:
      if not(self.cursorHV[0]):
        y0 = self.ylim[0]
        y1 = self.ylim[1]
      if not(self.cursorHV[1]):
        x0 = self.xlim[0]
        x1 = self.xlim[1]
    if self.rbPlotted:
      if self.GUIButton or self.GUICursor: # need to update plot
        if (self.GUICursor):
          d = 0.05
          self.rb.set_xdata([x0, x0, x1, x1, x0])
          self.rb.set_ydata([y0, y1, y1, y0, y0])
        else:           
          if self.xyGUI[0]:
            if (x0 < x1): 
              self.rb.set_xdata([x0, x0, x1, x1, x0])
            else:
              self.rb.set_xdata([self.xlim[0], self.xlim[0], self.xlim[1], self.xlim[1], self.xlim[0]])
          if self.xyGUI[1]:  
            if (y0 > y1):
              self.rb.set_ydata([y0, y1, y1, y0, y0])
            else:
              self.rb.set_ydata([self.ylim[0], self.ylim[1], self.ylim[1], self.ylim[0], self.ylim[0]])        
        self.fI.canvas.restore_region(self.bgbbox)
        self.aI.draw_artist(self.rb)
        self.fI.canvas.blit(self.aI.bbox)        
      else:      # need to delete plot
        scf(self.fI)
        mp.sca(self.aI)
        del self.aI.lines[len(self.aI.lines)-1]
        self.rbPlotted = 0 
        self.aI.axis(self.axxy) # [xmin, xmax, ymin, ymax]
        #self.fI.canvas.draw() -> no point drawing rubber-band initially for the first time
    else:        # need to construct a new plot  
      scf(self.fI)
      mp.sca(self.aI)
      self.bgbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox)
      if self.rbFormat == None or type(self.rbFormat) is str:
        self.rb, = mp.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], self.rbFormat, animated = True)   
      else:
        self.rb, = mp.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], color = self.rbFormat, animated = True)  
      self.rbPlotted = 1  
  def addGUIPlot(self, initFunc = None, calcFunc = None, evntList = [], funcList = [], plotStrg = ''):
    if type(calcFunc) is None:
      raise ValueError("calcFunc must be a [X,Y] list or a function that calculates it")
    self.plotsn += 1
    self.plotsInitialised.append(False) 
    self.plotsInitFunc.append(initFunc)
    self.plotsCalcFunc.append(calcFunc)  
    self.plotsEvntList.append(evntList)
    self.plotsFuncList.append(funcList) 
    self.plotsPlotStrg.append(plotStrg)
    self.GUIBypass = False 
    return self.plotsn - 1 
  def connectGUIPlot(self, evntList = [], funcList = []):
    ne, nf = len(evntList), len(funcList)
    if ne != nf: raise ValueError('Event specification list and function list incommensurate.')
    for i in range(ne):
      try:
        self.plotGUIKey[evntList[i]].append(funcList[i])  
      except KeyError:
        self.plotCID.append(self.fI.canvas.mpl_connect(evntList[i], funcList[i])) # Naughty
  def calcGUIPlot(self, plotid = None):
    x = None
    y = None
    if self.plotsInitialised[plotid] is False:
      self.plotsInitialised[plotid] = self.plotsInitFunc[plotid]()
      if self.plotsInitialised[plotid] is False: 
        return [x, y]      
    calcFunc = self.plotsCalcFunc[plotid]
    if type(calcFunc) is list:
      if len(calcFunc) == 1:
        y = calcFunc[0]
        x = np.arange(len(y))
      else:
        x, y = calcFunc
    elif not(calcFunc is None):
      [x, y] = calcFunc()      
    return [x,y]
  def drawGUIPlots(self, animated = None):
    # animated = None: Drawn for the first time
    # animated = False: On-mouse down or up
    # animated = True: On-mouse move 
    if not(self.plotsn): 
      self.draw(); 
      return
    if animated is None:
      if not(len(self.plots)):
        self.bgGUIbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox) 
        for i in range(self.plotsn):          
          self.connectGUIPlot(self.plotsEvntList[i], self.plotsFuncList[i])
          self.drawGUIPlot(i, True) 
          self.aI.draw_artist(self.plots[i])           
      elif not(self.origView):
        self.draw()
      return
    axw = ret_axes_width(self.fI, self.aI)     
    axh = ret_axes_height(self.fI, self.aI)     
    if self.plotaxwh is None:
      self.bgGUIbbox = None
    elif self.plotaxwh[0] != axw or self.plotaxwh[1] != axh:
      self.bgGUIbbox = None
    self.plotaxwh = [axw, axh]
    if animated:
      if self.bgGUIbbox is None: # cannot be animated if have no bgGUIbbox
        self.drawGUIPlots(False)
        return
      self.fI.canvas.restore_region(self.bgGUIbbox)  
      for i in range(self.plotsn):        
        self.drawGUIPlot(i, False)
        self.aI.draw_artist(self.plots[i])           
        self.fI.canvas.blit(self.aI.bbox)       
      return  
    if self.bgGUIbbox is None:
      for _plot in self.plots: _plot.set_animated(True)
      self.draw()           
      self.fI.canvas.blit(self.aI.bbox)        
      self.bgGUIbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox) 
      for _plot in self.plots: _plot.set_animated(False)
      self.draw() # plot needs to be drawn if animation not specified

    '''
    if animated is not True:
      if len(self.plots):
        for _plot in self.plots: 
          _plot.set_animated(True)
        self.draw()           
        if animated is False: self.fI.canvas.blit(self.aI.bbox)        
        self.bgGUIbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox) 
        for _plot in self.plots: 
          _plot.set_animated(False)
          if animated is None: self.aI.draw_artist(_plot) # drawing plots of the first time
        self.draw() # plot needs to be drawn if animation not specified
      else:  
        self.draw() # needs to be drawn for the first time before adding plots        
        self.bgGUIbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox) 
        for i in range(self.plotsn):          
          self.connectGUIPlot(self.plotsEvntList[i], self.plotsFuncList[i])
          self.drawGUIPlot(i, True) 
        self.draw()  
    else:
      self.fI.canvas.restore_region(self.bgGUIbbox)  
      for i in range(self.plotsn):        
        self.drawGUIPlot(i, False)
        self.aI.draw_artist(self.plots[i])           
        self.fI.canvas.blit(self.aI.bbox)       
    '''    
  def drawGUIPlot(self, plotid = None, newplot = False):
    if plotid is None: return
    if not(type(plotid is int)):
      _plotid = None
      for i in range(self.plotsn):
        if self.plots[i] ==  plotid: _plotid = i; i = self.plotsn
      plotid = _plotid  
    if newplot: # if drawn for the first time
      mp.sca(self.aI)            
      _plot = mp.plot(None, None, self.plotsPlotStrg[plotid])                   
      if len(_plot) == 1:
        self.plots.append(_plot[0])
      else:
        self.plots.append(_plot)     
    [x,y] = self.calcGUIPlot(plotid)        
    self.plots[plotid].set_xdata(x)
    self.plots[plotid].set_ydata(y)      
    return self.plots[plotid]          
  def update_axes(self, xmin = None, xmax = None, ymin = None, ymax = None):
    mp.sca(self.aI)
    if (xmin != None and xmax != None):
      self.aI.set_xlim(xmin, xmax)
    if (ymin != None and ymax != None): 
      self.aI.set_ylim(ymin, ymax)
    self.axxy = self.aI.axis() # [xmin, xmax, ymin, ymax]
    self.xlim = self.aI.get_xlim()
    self.ylim = self.aI.get_ylim()
    self.set_xlim(self.xlim[0], self.xlim[1])
    self.set_ylim(self.ylim[0], self.ylim[1])
  def set_xlim(self, x0 = None, x1 = None):
    if x0 == None:
      self.aI.set_xlim(self.origxlim)
      return
    dx = x1 - x0
    if x0 < self.origxlim[0]:
      x0 = self.origxlim[0]
      x1 = x0 + dx
    if x1 > self.origxlim[1]:
      x1 = self.origxlim[1]
      x0 = x1 - dx
    if x0 < self.origxlim[0]:
      x0 = self.origxlim[0]  
    self.aI.set_xlim(x0, x1)  
    self.xlim = self.aI.get_xlim()
    self.xran = self.xlim[1] - self.xlim[0]
    self.xmid = 0.5 * (self.xlim[0] + self.xlim[1])
    self.xwid = 0.5 * self.xran
    if self.yran is None: return
    self.xyrq = self.xran / unzero(self.yran)
    self.yxrq = 1./unzero(self.xyrq)
  def set_ylim(self, y0 = None, y1 = None):
    if y0 == None:
      self.aI.set_ylim(self.origylim)
      return    
    dy = y1 - y0
    if y0 < self.origylim[0]:
      y0 = self.origylim[0]
      y1 = y0 + dy
    if y1 > self.origylim[1]:
      y1 = self.origylim[1]
      y0 = y1 - dy
    if y0 < self.origylim[0]:
      y0 = self.origylim[0] 
    self.aI.set_ylim(y0, y1)  
    self.ylim = self.aI.get_ylim() 
    self.yran = self.ylim[1] - self.ylim[0]  
    self.ymid = 0.5 * (self.ylim[0] + self.ylim[1])
    self.ywid = 0.5 * self.yran 
    if self.xran is None: return   
    self.yxrq = self.yran / unzero(self.xran)
    self.xyrq = 1./unzero(self.yxrq)
  def zoomx(self, x0, x1):
    if not(self.xyGUI[0]): return
    if x0 < x1:
      self.set_xlim(x0, x1)
    #elif x0 > x1:
      #self.set_xlim()  
  def zoomy(self, y0, y1):    
    if not(self.xyGUI[1]): return
    if y0 > y1: # other way round for y-axis
      self.set_ylim(y1, y0)
    #elif y0 < y1:
      #self.set_ylim()
  def uzoomx(self, x, step):
    if not(self.xyGUI[0]): return
    mr = self.swSens * step
    if not(mr): mr = 1
    mr = -mr if mr < 0 else 1 / (mr)
    r = mr * self.xran 
    self.set_xlim(x - r/2 , x + r/2)
  def uzoomy(self, y, step):
    if not(self.xyGUI[1]): return
    mr = self.swSens * step
    if not(mr): mr = 1
    mr = -mr if mr < 0 else 1 / (mr)
    r = mr * self.yran 
    self.set_ylim(y - r/2 , y + r/2)  
  def pan(self):
    if (self.xyGUI[2]) :
      dx = (self.xend - self.xbeg)
      xl = self.aI.get_xlim()
      self.aI.set_xlim(xl[0]-dx, xl[1]-dx)
    if (self.xyGUI[3]) : 
      dy = (self.yend - self.ybeg)
      yl = self.aI.get_ylim()
      self.aI.set_ylim(yl[0]-dy, yl[1]-dy)
    self.fI.canvas.draw()     
  def panx(self, x0, x1):    
    if not(self.xyGUI[2]): return
    if self.origxlim == None: return
    if x0 < self.origxlim[0]:
      x0 = self.origxlim[0]
      x1 = x0 + self.xran
    if x1 > self.origxlim[1]:
      x1 = self.origxlim[1]
      x0 = x1 - self.xran 
    self.aI.set_xlim(x0, x1)      
  def pany(self, y0, y1):    
    if not(self.xyGUI[3]): return
    if self.origylim == None: return
    if y0 < self.origylim[0]:
      y0 = self.origylim[0]
      y1 = y0 + self.yran
    if y1 > self.origylim[1]:
      y1 = self.origylim[1]
      y0 = y1 - self.yran 
    self.aI.set_ylim(y0, y1)     
  def relDist(self, x0, y0, x1 = None, y1 = None):
    if x1 is None: x1 = self.xmid   
    if y1 is None: y1 = self.ymid
    return np.sqrt( ((x0 - x1)/self.xran)**2. + ((y0 - y1)/self.yran)**2. )
  def draw(self):
    self.fI.canvas.draw()
    xlims = self.aI.get_xlim()
    ylims = self.aI.get_ylim()    
  def redraw(self, resize = False, forcedraw = False, event = None):
    if not(figexists(self.fI)): return
    if self.plotsn:
      self.bgGUIbbox = None # this forces GUI-box recapture following resizing
    if resize: # this calls any relevant resize functions which may include this one   
      self.fI.canvas.resize_event()
      if event is not None: return
    scf(self.fI)
    try: 
      mp.sca(self.aI)    
    except ValueError: # axis does not exist
      return
    mp.ioff()
    xlims = self.aI.get_xlim()
    ylims = self.aI.get_ylim()    
    self.update_axes()
    notdraw = not(forcedraw or self.origView)
    if notdraw:  
      notdraw = (self.xlim[0] == xlims[0] and self.xlim[1] == xlims[1]) and (self.xlim[0] == xlims[0] and self.xlim[1] == xlims[1])
    if not(notdraw):
      if not(self.origView): 
        self.drawGUIPlots()
      self.bgbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox)       
    if self.isint:
      mp.ion()
  def on_resize_event(self, event):
    self.redraw(False, True, event)  
  def on_draw_event(self, event):
    if not(self.origView): return
    self.update_axes()          
    if self.origView:
      self.origxlim = self.xlim
      self.origylim = self.ylim   
    self.origView = False
    self.redraw(True, True, event)  
  def on_button_press_event(self, event):
    self.update_axes()  
    self.iaI.refreshAxes()    
    for plotBPEfunc in self.plotBPEFuncList:
      if plotBPEfunc is not None:
        if plotBPEfunc(event): 
          self.GUIBypass = True
      if self.GUIBypass: return
    self.inMotion = False      
    self.xbeg = event.xdata 
    self.ybeg = event.ydata
    self.xend = event.xdata 
    self.yend = event.ydata 
    self.GUIButton = event.button  
    if (self.GUICursor and self.cursorCBFunc != self.cursorPair): return # deal with cursor pairs here    
    _GUICursor = self.GUICursor
    if _GUICursor or self.GUIButton == LEFTBUTTON:      
      if _GUICursor:
        self.setCursor() # remove cursor
      self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend)  
      if _GUICursor:
        self.cursorCBFunc(event) # to re-establish new cursor
  def on_motion_notify_event(self, event):    
    if self.GUIBypass:
      self.iaI.boundEvent(event)
      for plotMNEfunc in self.plotMNEFuncList:
        if plotMNEfunc is not None: plotMNEfunc(event)
      return        
    if not(self.GUIButton): return
    if (self.GUICursor and not(self.curscounter)): return
    if not(self.iF.button_pressed):
      self.GUIButton = 0
      return
    self.iaI.boundEvent(event)
    self.inMotion = True    
    self.xend = event.xdata
    self.yend = event.ydata    
    if (self.GUICursor):
      self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend)    
    elif self.GUIButton == LEFTBUTTON:
      self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend)
    elif self.GUIButton == MIDDLEBUTTON:
      self.pan()
  def on_button_release_event(self, event):    
    if self.GUIBypass:
      self.iaI.boundEvent(event)
      for plotBREfunc in self.plotBREFuncList:
        if plotBREfunc is not None: plotBREfunc(event)
      self.GUIBypass = False  
      return        
    if not(self.GUIButton): return # we tolerate releasing out-of-axis
    self.iaI.boundEvent(event)
    _btn = self.GUIButton
    self.GUIButton = 0
    if self.GUICursor:
      if (self.cursorCBFunc == self.cursorPair):
        self.cursorPair(event)
        self.redraw(False, True) # in case GUI disconnects
        return
      self.setCursor()
      self.cursorCBFunc(event)
      return
    if not(self.inMotion): # if the mouse was still
      if _btn == LEFTBUTTON: # remove RB Plot        
        self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend)        
      for stillclickfunc in self.stillClickFunc:
        if stillclickfunc is not None: stillclickfunc(event)
      return      
    if (event.inaxes != self.aI): # user has crossed two plots
      self.update_axes() 
    else:  
      self.xend = event.xdata
      self.yend = event.ydata       
    if _btn == LEFTBUTTON:  
      self.plotRB(self.xbeg, self.ybeg, self.xend, self.yend)  
      if self.xbeg > self.xend and self.ybeg < self.yend:
        self.set_xlim()
        self.set_ylim()
      else:  
        self.zoomx(self.xbeg, self.xend)
        self.zoomy(self.ybeg, self.yend) 
    elif _btn == MIDDLEBUTTON:
      xl = self.aI.get_xlim()
      yl = self.aI.get_ylim()
      self.panx(xl[0], xl[1])
      self.pany(yl[0], yl[1])
    if self.inMotion: self.redraw(True, True)
  def on_scroll_event(self, event):
    if event.inaxes != self.aI or not(self.xyGUI[4]) or self.GUICursor: 
      return;
    self.update_axes()
    if self.xyGUI[4] == 1: # zoom
      self.uzoomx(event.xdata, float(event.step))
      self.uzoomy(event.ydata, float(event.step))
    elif self.xyGUI[4] == 2: # pany  
      dy = self.swSens * event.step * self.yran / 100.0
      self.pany(self.ylim[0] + dy, self.ylim[1] + dy)
    self.redraw(True)  
  def on_key_press_event(self, event):
    for plotKPEfunc in self.plotKPEFuncList:
      if plotKPEfunc is not None: plotKPEfunc(event)
  def on_pick_event(self, event):
    for plotPEfunc in self.plotPEFuncList:
      if plotPEfunc is not None: plotPEfunc(event)

class itxtplot:
  defxypad = [0.05, 0.05]
  plotset = 0
  fI = None
  cidbpe = None
  def __init__(self, argIn0 = None, argIn1 = None, argIn2 = None):
    self.initialise()
    self.subplot(argIn0, argIn1, argIn2)
  def initialise(self, _data = None, _mdcf = None):
    self.setPad()
    self.setData(_data, _mdcf)    
  def setData(self, _data = None, _mdcf = None, _fontsize = None):
    self.data = _data 
    self.mdcf = _mdcf
    if self.data is None: return
    if not(type(self.data) is list):
      raise ValueError('Data type must be a 2D list')
    if nDim(self.data) != 2:
      raise ValueError('Data type must be a 2D list')
    self.ncols = nCols(self.data)
    self.nrows = nRows(self.data)
    if self.plotSet:
      self.connectGUI(self.fI, _mdcf)
      self.drawTable(_fontsize)
  def setPad(self, _padx = None, _pady = None):
    self.padx = _padx if _padx != None else self.defxypad[0]
    self.pady = _pady if _pady != None else self.defxypad[1]
  def subplot(self, argIn0 = None, argIn1 = None, argIn2 = None):
    self.aI = None
    if argIn2 != None:
      self.aI = mp.subplot(argIn0, argIn1, argIn2)
      self.plotSet = 1
    elif argIn0 != None:
      self.aI = mp.subplot(argIn0)
      self.plotSet = 1
    elif argIn1 == None:
      self.aI = mp.gca()
      self.plotSet = 1
    if self.plotSet:
      self.connectGUI(self.aI.figure, self.mdcf)
    return self.aI
  def connectGUI(self, _fI = None, _mdcf = None):
    self.fI = _fI
    self.mdcf = _mdcf    
    if self.fI is None: return
    self.canv = self.fI.canvas
    if not(self.mdcf is None):
      self.cidbpe = self.canv.mpl_connect("button_press_event", self.mdcf)
  def drawTable(self, _fontsize = None):
    if not(self.ncols) or not(self.nrows): return
    if (self.fI is None or self.aI is None): return
    scf(self.fI)
    mp.sca(self.aI)
    self.x = np.linspace(self.padx, 1.0-self.padx, self.ncols+1)
    self.y = np.linspace(self.pady, 1.0-self.pady, self.nrows)
    self.x = self.x[:-1]
    self.y = self.y[::-1]
    mp.hold(True)
    for i in range(self.nrows):
      for j in range(self.ncols):
        if _fontsize == None:
          mp.text(self.x[j], self.y[i], str(self.data[i][j]), verticalalignment='center')
        else:
          mp.text(self.x[j], self.y[i], str(self.data[i][j]), fontsize = _fontsize, verticalalignment='center')
    mp.xlim([0, 1])
    mp.ylim([0, 1])
    showticks(self.aI, False, False)
  def event2index(self, event = None):
    if event.inaxes != self.aI: return [None, None]
    dx = event.xdata - self.x
    dy = event.ydata - self.y
    dx[dx < 0] = np.inf
    dy[dy < 0] = np.inf
    return np.argmin(dx), np.argmin(dy)

class iscatter (isubplot):
  Npick = 0
  mbtn = 0
  Ind = None
  ind = None
  pickevent = None
  picked = False
  moved = False
  nGUIEllipse = 0
  pickP = None
  def __init__(self, argIn0 = None, argIn1 = None, argIn2 = None):
    self.initialise()
    self.subplot(argIn0, argIn1, argIn2)
    self.setPicker()
    self.ellGUI = -1
    self.ell = None
    self.ellFunc = []
    self.scatP = []
  def addGUIEllipse(self, ellfunc = None, ellstr = '', ell2str = 'o', ellGUIthickness = [0.03, 0.04]):
    self.nGUIEllipse += 1
    self.ellFunc.append(ellfunc)
    self.ellGUIThickness = ellGUIthickness
    self.ellipGUI = -1 # -1 = Not moved, 0->1 moving points x0,y0 or x1,y1, 2 = moving ellipse centre
    buttonstr = ["button_press_event", "motion_notify_event", "button_release_event"]
    functions = [self.on_mouse_down, self.on_mouse_move, self.on_mouse_up]
    self.elplotid = self.addGUIPlot(self.initGUIEllipse, self.calcGUIEllipse, buttonstr, functions, ellstr)  
    self.e2plotid = self.addGUIPlot(self.initGUIEllips2, self.calcGUIEllips2, [], [], ell2str)    
  def initGUIEllipse(self, _xyabps = None, _ellipr = None, _ellipn = None):    
    self.ell = GUIEllipse(_xyabps, _ellipr, _ellipn)
    self.ellipn = self.ell.n
    self.elli4i = np.array([0, 0.25*self.ellipn, 0.5*self.ellipn, 0.75*self.ellipn], dtype = int)    
    if _xyabps is None:
      if (self.xmid is None or self.ymid is None) or (self.xwid is None or self.ywid is None):
        return False # failed initialisation
      _xyabps = [[]]*6
      _xyabps[0] = self.xmid
      _xyabps[1] = self.ymid
      _xyabps[2] = self.xwid
      _xyabps[3] = self.ywid
      _xyabps[4] = 0.
      _xyabps[5] = 1.
      self.ell.set_xyabps(_xyabps)
    if _ellipr is None:  
      _ellipr = 0.5 
    self.ell.set_r(_ellipr)    
  def initGUIEllips2(self):
    self.elli2x = np.empty(2, dtype = float)
    self.elli2y = np.empty(2, dtype = float)  
  def calcGUIEllipse(self):    
    self.ellipx, self.ellipy = self.ell.calcEllipse()
    return [self.ellipx, self.ellipy]
  def calcGUIEllips2(self):  
    x4, y4 = self.ellipx[self.elli4i], self.ellipy[self.elli4i]
    if self.origView:
      self.elli2x[0], self.elli2y[0] = x4[2], y4[2]
      self.elli2x[1], self.elli2y[1] = x4[1], y4[1]
    else:  
      z4 = self.relDist(x4, y4)
      if z4[0] < z4[2]:
        self.elli2x[0], self.elli2y[0] = x4[0], y4[0]
      else:
        self.elli2x[0], self.elli2y[0] = x4[2], y4[2]
      if z4[1] <= z4[3]:
        self.elli2x[1], self.elli2y[1] = x4[1], y4[1]
      else:
        self.elli2x[1], self.elli2y[1] = x4[3], y4[3]       
    return [self.elli2x, self.elli2y]
  def setPicker(self, _pickerstr = 'o', _pickersize = 8, _pickerfunc = None):
    self.pickerStr = _pickerstr
    self.pickerSize = _pickersize
    self.pickerFunc = _pickerfunc
  def unpickAll(self):
    self.Ind = np.array([], dtype = int)
    self.ind = np.array([], dtype = int)
    self.xpick = [[]] * self.scatN
    self.ypick = [[]] * self.scatN
    self.npick = [0] * self.scatN    
    if self.pickP is None:
      self.pickP = [None] * self.scatN
      self.mecpk = [[]] * self.scatN
      self.mfcpk = [[]] * self.scatN      
    else:
      for i in range(self.scatN):
        if self.pickP[i] is not None:
          self.pickP[i].set_xdata(None)
          self.pickP[i].set_ydata(None)          
    self.Npick = 0
  def plot(self, _x, _y = None, *args, **kwargs):
    if _y == None: _y, _x = np.copy(_x), np.arange(len(_x))
    self.scatx = _x
    self.scaty = _y
    self.scatp = self.aI.plot(self.scatx, self.scaty, *args, **kwargs)  
    if len(self.scatp) == 1: self.scatp = self.scatp[0]
    self.scatP.append(self.scatp)
    self.scatN = len(self.scatP)
    mp.setp(self.scatp, picker = self.pickerSize)
    if len(self.scatP) == 1: 
      self.addStillClickFunc(self.on_picked)
    return self.scatp  
  def calcInd(self, event):
    _Ind = None
    _ind = None
    self.scatp = event.artist
    for i in range(self.scatN):
      if self.scatP[i] == self.scatp:
        _Ind = i;
        _ind = event.ind
    return _Ind, _ind   
  def pick(self, _Ind, _ind):
    unpick = False
    if type(_ind) is np.ndarray:
      if _ind.size == 1:
        _ind = int(_ind)   
    if self.Npick == 1:
      if type(_Ind) is int and type(_ind) is int:
        unpick = self.Ind == _Ind and self.ind == _ind  
    if type(_Ind) is int: _Ind = [_Ind]
    if type(_ind) is int: _ind = [_ind]
    self.unpickAll();
    if unpick: self.plotpick; return        
    self.Ind = np.array(_Ind, dtype = int)
    self.ind = np.array(_ind, dtype = int)
    if self.ind.size == 1 and self.ind.size < self.Ind.size:
      self.ind = np.tile(self.ind, self.Ind.size)
    elif self.Ind.size == 1 and self.Ind.size < self.ind.size:  
      self.Ind = np.tile(self.Ind, self.ind.size)
    for i in range(self.scatN):
      indi = np.nonzero(i == self.Ind)[0]
      self.npick[i] = len(indi)
      self.Npick += self.npick[i]
      if self.npick[i]: 
        self.xpick[i] = self.scatP[i].get_xdata()[self.ind[indi]]
        self.ypick[i] = self.scatP[i].get_ydata()[self.ind[indi]]
        self.mecpk[i] = self.scatP[i].get_markeredgecolor()
        self.mfcpk[i] = self.scatP[i].get_markerfacecolor()
      else:  
        self.mecpk[i] = None
        self.mfcpk[i] = None
    self.plotpick()      
  def plotpick(self):
    if not(self.Npick): return
    scf(self.fI)
    mp.sca(self.aI)    
    for i in range(self.scatN):
      if self.pickP[i] is None:
        self.pickP[i], = mp.plot(self.xpick[i], self.ypick[i], self.pickerStr)
      else:  
        self.pickP[i].set_xdata(self.xpick[i])
        self.pickP[i].set_ydata(self.ypick[i])
      self.pickP[i].set_markeredgecolor(self.mecpk[i])  
      self.pickP[i].set_markerfacecolor(self.mfcpk[i])
      if self.npick[i]: self.aI.draw_artist(self.pickP[i])
    self.fI.canvas.blit(self.aI.bbox)     
    self.bgbbox = self.fI.canvas.copy_from_bbox(self.aI.bbox) 
  def on_mouse_down(self, event = None):
    if (event.inaxes != self.aI): return False
    self.moved = False
    if (self.nGUIEllipse < 1): return False
    if not(event.xdata is None): self.xbeg = event.xdata 
    if not(event.ydata is None): self.ybeg = event.ydata  
    self.mbtn = event.button
    self.xyabps = self.ell.xyabps[:]
    self.ellipr = self.ell.r
    reldis = self.relDist(self.elli2x, self.elli2y, self.xbeg, self.ybeg)
    if reldis[0] <= self.ellGUIThickness[1]: 
      self.drawGUIPlots(False)
      self.ellGUI = 0;            
      return True
    if reldis[1] <= self.ellGUIThickness[1]: # invert major and minor axes
      self.drawGUIPlots(False)
      Dx = (self.elli2x[1] - self.ell.xyabps[0]) / unzero(self.ell.xyabps[2])
      Dy = (self.elli2y[1] - self.ell.xyabps[1]) / unzero(self.ell.xyabps[3])
      self.ell.r = np.sqrt(Dx**2. + Dy**2.)
      Dx = (self.elli2x[0] - self.ell.xyabps[0]) / unzero(self.ell.xyabps[2])
      Dy = (self.elli2y[0] - self.ell.xyabps[1]) / unzero(self.ell.xyabps[3])
      self.ell.xyabps[4] += 0.5*np.pi
      self.ell.xyabps[5] = np.sqrt(Dx**2. + Dy**2.) / unzero(self.ell.r)
      self.xyabps = self.ell.xyabps[:]
      self.ellipr = self.ell.r      
      self.ellGUI = 1;
      return True
    reldis = self.relDist(self.ellipx, self.ellipy, self.xbeg, self.ybeg)
    if np.min(reldis) <= self.ellGUIThickness[0]: 
      self.drawGUIPlots(False)
      self.ellGUI = 2; 
      return True
    return False
  def on_mouse_move(self, event = None):
    #self.iaI.boundEvent(event)
    self.moved = True    
    if self.mbtn != LEFTBUTTON: # moved while using other buttons: demote to pan
      self.ellGUI = -1;
    if (self.ellGUI < 0): return
    self.picked = False # this prevents inadvertant picking
    if not(event.xdata is None): self.xend = event.xdata 
    if not(event.ydata is None): self.yend = event.ydata
    if self.ellGUI == 0 or self.ellGUI == 1:
      Dx = (self.xend - self.xyabps[0]) / self.xyabps[2]
      Dy = (self.yend - self.xyabps[1]) / self.xyabps[3]
      self.ell.r = np.sqrt(Dx**2 + Dy**2)
      self.ell.xyabps[4] = atan2(Dx, Dy)
      self.ell.xyabps[5] = self.xyabps[5] * self.ellipr / unzero(self.ell.r)
    elif self.ellGUI == 2:
      self.ell.xyabps[0] = self.xyabps[0] + self.xend - self.xbeg
      self.ell.xyabps[1] = self.xyabps[1] + self.yend - self.ybeg    
    self.drawGUIPlots(True) # we animate the ellipse only on movement
  def on_mouse_up(self, event = None):  # this pertains only to ellipses (picking is a still-click)
    if (self.ellGUI < 0 and not(self.picked)): 
      return 
    _ellGUI = self.ellGUI    
    self.ellGUI = -1    
    if self.moved or self.mbtn != LEFTBUTTON: self.picked = False # this prevents inadvertant picking
    if not(self.moved): 
      if _ellGUI >= 0 and self.mbtn == RIGHTBUTTON:
        self.initGUIEllipse(None, None, self.ellipn)
        self.drawGUIPlots(True) 
      elif self.picked and self.mbtn == LEFTBUTTON: # override ellipse
        self.on_picked(event) # to be sure it is called
        return    
    if len(self.ellFunc):  
      for ellfunc in self.ellFunc:
        if ellfunc is not None: 
          ellfunc(event)      
      self.bgGUIbbox = None
    self.drawGUIPlots(False) 
  def on_pick_event(self, event):
    self.pickevent = event
    self.picked = True    
  def on_picked(self, event):
    if event.inaxes != self.aI: picked = False; return
    if event.button != LEFTBUTTON: picked = False; return
    if not(self.picked): return
    self.ellGUI = -1 
    self.mbtn = 0 # this attempts to suppress ellipse movement
    self.picked = False    
    _Ind, _ind = self.calcInd(self.pickevent)
    if _Ind is None or _ind is None: return
    self.pick(_Ind, _ind)
    bypass = False
    if self.pickerFunc is not None: 
      if (self.pickerFunc(event)): bypass = True
    if not(bypass): self.drawGUIPlots(False)
    return 

