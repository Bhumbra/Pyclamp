import os
import sys
import platform
import numpy as np
from time import time
from pyclamp.dsp.lsfunc import *
from pyclamp.dsp.dtypes import *
from pyclamp.gui.display import DISPLAY

PLATFORM = platform.platform()
CONSOLECOL = {'w':97, 'm':95, 'r':91, 'y':93, 'g':92, 'c':96, 'b':94, 'k':30, 'e':1,
              'W':97, 'M':95, 'R':91, 'Y':93, 'G':92, 'C':96, 'B':94, 'K':30, 'E':1}

if DISPLAY is None:
  QtGui = None
  QAPP = None
else:
  from PyQt4 import QtGui
  QAPP = QtGui.QApplication

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

class cpgb (object):      # console progressbar class for console outputs with multiple bar support
  mino = 1e-300
  mindt = 0.03   # secs
  tmin = 0.0
  N = 0
  n = 0
  reprint = True
  rw = 1
  defcw = 50     # default column (`virtual') width if tty unavailable
  ldd = 0        # last denominator decremented
  st = None      # console string
  cwpt = 0.3     # polling interval to check console column width (secs)
  def __init__(self, _pgbtitle = None, _tit = 'Please wait...', _den = 100, _col = 'b'):
    if isint(_tit): # deal with single-bar case
      if type(_den) is str: # 3 input arguments
        _pgbtitle, _tit, _den, _col= None, _pgbtitle, _tit, _den
      else:                 # 2 input arguments
        _pgbtitle, _tit, _den = None, _pgbtitle, _tit
    self.settitle(_pgbtitle)
    self.init(_tit, _den, _col)
    self.setbars()
    self.forceupdate() # ensure first update
  def init(self, _tit = 'Please wait...', _den = 100, _col = 'b'):
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
    self.setbars()
    self.forceupdate() # ensure first update
  def setbars(self):
    if not(self.N): return
    if (self.N != self.n):
      self.addbars()
    self.updatetitles()
    self.set([0] * self.n)
  def addbars(self):
    self.N = self.n
  def setlast(self, _titlast = 'Please wait', _denlast = 100, _collast = 'b'):
    self.tit[-1] = _titlast
    self.den[-1] = 0.01 * float(_denlast)
    self.dnd[-1] = _denlast - 1
    self.col[-1] = _collast
    self.num[-1] = 0.0
    self.pct[-1] = -1
    self.updatelast(0)
  def forceupdate(self, _num = None):
    self.tmin = time() - self.mindt
    if not(_num is None): self.set(_num, True)
  def set(self, _num, _redraw = False):
    if not(type(_num) is list): return self.updatelast(_num, _redraw)
    t = time()
    redraw = _redraw or _num[-1] == self.dnd[-1]
    if t < self.tmin and not(redraw): return
    if redraw:
      self.pct[-1] = 100
      self.updatebars(self.n-1)
      self.tmin = t - self.mindt # force a next update
      self.draw()
      return
    self.tmin = t + self.mindt
    for i in self.I:
      if _num[i] is not None:
        _pct = round(float(_num[i]) / (self.mino + self.den[i]))
        if self.pct[i] != _pct:
          redraw = True
          self.pct[i] = int(_pct)
          self.updatebars(i)
          if i < self.n - 1:
            self.reprint = 1
    if redraw:
      self.draw()
  def updatelast(self, _numlast, _redraw = False):
    self.non[-1] = _numlast
    self.set(self.non, _redraw)
  def close(self):
    print('\n')
    self.N = 0
  def reset(self):
    self.updatelast(self.dnd[-1])
    return self.close
  def settitle(self, _pgbtitle = None):
    self.pgbtitle = _pgbtitle
    self.cpgbtitle = self.pgbtitle
    if self.pgbtitle is not None:
      self.cpgbtitle = colstr(self.pgbtitle, 'e')
  def setbars(self):
    if (self.N != self.n):
      self.addbars()
    self.updatetitles()
    self.set([0] * self.n)
  def updatetitles(self):
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
  def updatebars(self, i):
    t = time()
    if t - self.cwt >= self.cwpt:
      self.cwt = t
      if self.rw is not None:
        _, _cw = ttysize()
        if self.cw != _cw: self.updatetitles()
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
  def draw(self):
    if self.reprint:
      if self.rw is None:
        sys.stdout.write('\n')
        if self.pgbtitle is not None:
          sys.stdout.write(self.pgbtitle)
          sys.stdout.write('\n')
        for i in self.I[:-1]:
          sys.stdout.write(self.st[i])
      else:
        if self.cpgbtitle is not None:
          print('\n')
          print(self.pgbtitle)
        else:
          print('\r')
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

class qpgb (object):
  mindt = 0.05 # secs
  maxdt = 0.5 # secs
  cantext = "Cancel"
  tmin = np.inf
  minm = 0.
  maxm = 100.
  maxd = 99.
  step = 1.
  lomx = False
  dlg = None
  def __init__(self, _title = "Processing...", _maxm = 100., _cancelfunc = None):
    self.init(_title, _maxm, _cancelfunc)
  def init(self, _title = "Processing...", _maxm = 100., _cancelfunc = None):
    self.title = _title
    self.minm = 0.
    self.maxm = _maxm
    self.maxd = self.maxm-1
    self.step = int(max(1, float(self.maxm) / 100.))
    self.lomx = self.maxm <= 100
    self.cancelfunc = _cancelfunc
    '''
    self.prc = pgmp.QtProcess()
    self.rqt = self.prc._import('PyQt4.QtGui')
    self.dlg = self.rqt.QProgressDialog(self.title, self.cantext, self.minm, self.maxm)
    '''
    self.dlg = QtGui.QProgressDialog(self.title, self.cantext, self.minm, self.maxm)
    #if self.cancelfunc is None: self.dlg.setCancelButton(0)
    self.set(0) # for some reason a zero-call is needed to kick-start it to appear
    self.show()
  def show(self):
    if self.dlg is None: return
    self.dlg.show()
  def set(self, x = 1, *args):
    t = time()
    if self.lomx or x == 0 or x == self.maxd:
      try:
        self.dlg.setValue(x, _callSync='off')
      except TypeError:
        self.dlg.setValue(x)
    elif t < self.tmin:
      return
    else:
      if t < self.tmax:
        if x % self.step:
          return
      try:
        self.dlg.setValue(x, _callSync='off')
      except TypeError:
        self.dlg.setValue(x)
    self.tmin = t + self.mindt
    self.tmax = t + self.maxdt
    if self.cancelfunc is None: return
    canceled = self.dlg.wasCanceled()
    if canceled:
      self.cancelfunc()
  def close(self, *args):
    try:
      self.dlg.reset()
    except AttributeError:
      pass
  def reset(self, *args):
    self.close()

class pgb (object):
  defobj = None
  defsetfunc = None
  defresetfunc = None
  defclosefunc = None
  GUIobj = qpgb
  GUIsetfunc = qpgb.set
  GUIresetfunc = qpgb.close
  GUIclosefunc = qpgb.close
  TTYobj = cpgb
  TTYresetfunc = cpgb.reset
  TTYsetfunc = cpgb.set
  TTYclosefunc = cpgb.close
  def __init__(self, _obj = None, *args, **kwds):
    self.defGUI = DISPLAY is not None
    if type(_obj) is str: # A `title' bypasses object specification
      self.setObj()
      self.setFunc()
      self.init(_obj, *args, **kwds)
      return
    self.setObj(_obj)
    self.setFunc(*args)
  def setObj(self, _obj = None):
    if _obj is None: _obj = self.defGUI
    if type(_obj) is bool:
      if _obj and DISPLAY:
        self.defobj = self.GUIobj
        self.defsetfunc = self.GUIsetfunc
        self.defresetfunc = self.GUIresetfunc
        self.defclosefunc = self.GUIclosefunc
      else:
        self.defobj = self.TTYobj
        self.defsetfunc = self.TTYsetfunc
        self.defresetfunc = self.TTYresetfunc
        self.defclosefunc = self.TTYclosefunc
      _obj = None
    self.obj = self.defobj if _obj is None else _obj
  def setFunc(self, _setfunc = None, _resetfunc = None, _closefunc = None):
    self.setfunc = self.defsetfunc if _setfunc is None else _setfunc
    self.resetfunc = self.defresetfunc if _resetfunc is None else _resetfunc
    self.closefunc = self.defclosefunc if _closefunc is None else _closefunc
  def init(self, *args, **kwds):
    nargs = len(args) if self.obj == self.GUIobj else 0
    if nargs > 1:
      listip = False
      for i in range(nargs):
        if type(args[i]) is list:
          listip = True
      if listip:
        self.setObj(False)
        self.setFunc()
    self.Obj = self.obj(*args, **kwds)
    return self.Obj
  def set(self, *args, **kwds):
    #return self.setfunc(self.Obj, *args, **kwds)
    return self.Obj.set(*args, **kwds)
  def reset(self, *args, **kwds):
    #return self.resetfunc(self.Obj, *args, **kwds)
    return self.Obj.reset(*args, **kwds)
  def close(self, *args, **kwds):
    #self.Out = self.closefunc(self.Obj, *args, **kwds)
    self.Out =  self.Obj.close(*args, **kwds)
    del self.Obj
    return self.Out

