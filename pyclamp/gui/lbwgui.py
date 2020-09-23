#!/usr/bin/env python

import sys
import os
from pyclamp.dsp.strfunc import isnumeric
from pyclamp.dsp.lsfunc import *

DEFLBWUSE = 'qt'

def use(_LBWUSE = None):
  if _LBWUSE is None:
    _LBWUSE = DEFLBWUSE
  global LBWUSE
  try:
    lbwuse = LBWUSE # force error for when setting default
    LBWUSE = _LBWUSE.lower()
  except NameError:
    LBWUSE = _LBWUSE.lower()

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets

class qtKeys: # code to handle Qwidget-specific functions
  Wid = None
  Widget = None
  FWidget = None
  Parent = None
  def __init__(self):
    self.widget = {'base': QtWidgets.QWidget, 'panel': QtWidgets.QFrame, 'label': QtWidgets.QLabel, 'edit': QtWidgets.QLineEdit,
        'listbox':QtWidgets.QListWidget, 'combobox': QtWidgets.QComboBox, 'button':QtWidgets.QPushButton, 'box': QtWidgets.QBoxLayout,
        'groupbox': QtWidgets.QBoxLayout, 'radiobutton':QtWidgets.QRadioButton, 'buttongroup': QtWidgets.QButtonGroup, 'labelbox':
        QtWidgets.QGroupBox, 'frame': QtWidgets.QFrame, 'dlgmess':QtWidgets.QMessageBox, 'dlgform':
        QtWidgets.QDialog, 'form': QtWidgets.QMainWindow, 'mainform': QtWidgets.QMainWindow, 'childarea':QtWidgets.QMdiArea,
        'childform':QtWidgets.QMdiSubWindow, 'app': QtWidgets.QApplication}
    self.nargin1 = ['box', 'groupbox']
    self.nargin2 = ['checkbox']
    self.direction = {0 : QtWidgets.QBoxLayout.LeftToRight, 1 : QtWidgets.QBoxLayout.TopToBottom}
  def setParent(self, parent = None):
    if parent is not None: self.Parent = parent
  def setChild(self, child = None):
    if child is None: return
    if self.Wid != 'mainform': return
    self.FWidget.setCentralWidget(child)
  def addChild(self, label = ''):
    return self.Widget.addSubWindow(QtWidgets.QLabel(label))
  def setWidget(self, wid = None, *args):
    if (wid is None): return
    self.Wid = wid
    if len(args):
      if self.Wid == 'box' or self.Wid == 'groupbox':
        self.Widget = self.widget[self.Wid](self.boxDir(args[0]))
      else:
        self.Widget = self.widget[self.Wid](args)
    else:
      if self.Wid == 'app':
        self.Widget = self.widget[self.Wid](sys.argv)
      elif self.Wid.count("form"):
        self.FWidget = self.widget[self.Wid]()
        if self.Wid == "mainform":
          self.Widget = self.widget['base']()
          self.FWidget.setCentralWidget(self.Widget)
        else:
          self.Widget = self.FWidget
      else:
        if self.Wid == 'box' or self.Wid == 'groupbox':
          self.Widget = self.widget[self.Wid](self.boxDir(0))
        elif self.Wid == 'buttongroup':
          self.Widget = self.widget[self.Wid]()
          self.FWidget = self.widget['labelbox']()
        else:
          self.Widget = self.widget[self.Wid]()
      return self.Widget, self.FWidget
  def setData(self, data):
    if self.Wid is None or self.Widget is None or data is None: return
    if self.Wid == 'label':
      self.Widget.setText(data)
    if self.Wid == 'edit':
      self.Widget.setText(data)
    elif self.Wid == 'labelbox':
      self.Widget.setTitle(data)
    elif self.Wid == 'button':
      self.Widget.setText(data)
    elif self.Wid == "listbox":
      if type(data) is tuple or type(data) is list or type(data) is range:
        for _data in data:
          self.Widget.item(_data).setSelected(True)
      else:
        self.Widget.setCurrentRow(int(data))
    elif self.Wid == "combobox":
      self.Widget.setCurrentIndex(int(data))
    elif self.Wid == "dlgmess":
      self.Widget.setText(data)
    elif self.Wid == "radiobutton":
      self.Widget.setDown(data)
      if type(data) is bool:
        self.Widget.setChecked(data)
    elif self.Wid == "buttongroup":
      if type(data) is list:
        i = 0
        isExclusive = self.Widget.exclusive()
        self.Widget.setExclusive(False)
        for btn in self.Widget.buttons():
          btn.setChecked(data[i])
          i += 1
        self.Widget.setExclusive(isExclusive)
      else:
        self.FWidget.setTitle(data)
    elif self.Wid == "dlgform":
      self.FWidget.setWindowTitle(data)
    elif self.Wid == "mainform":
      self.FWidget.setWindowTitle(data)
  def setOpts(self, opts):
    if self.Wid is None or self.Widget is None or opts is None: return
    if self.Wid == 'listbox':
      self.Widget.insertItems(0, opts)
    elif self.Wid == 'combobox':
      self.Widget.insertItems(0, opts)
    elif self.Wid == "radiobutton":
      self.Widget.setText(opts)
  def retData(self):
    if self.Wid is None or self.Widget is None: return None
    if self.Wid == "edit":
      return str(self.Widget.text())
    elif self.Wid == "listbox":
      outData = []
      for item in self.Widget.selectedItems():
        outData.append( (self.Widget.indexFromItem(item)).row())
      return outData
    elif self.Wid == "combobox":
      return self.Widget.currentIndex()
    elif self.Wid == "buttongroup":
      outData = self.Widget.checkedId()
      if outData != -1:
        outData = -outData - 2
    elif self.Wid == "radiobutton":
      return self.Widget.isChecked()
    return outData
  def addWidget(self, child):
    if self.Wid == 'buttongroup':
      self.Widget.addButton(child)
    else:
      self.Widget.addWidget(child)
  def setBox(self, child):
    if self.Wid == "buttongroup":
      self.FWidget.setLayout(child)
    else:
      self.Widget.setLayout(child)
  def addBox(self, child):
    self.Widget.addLayout(child)
  def addSpace(self, space):
    self.Widget.addSpace(space)
  def boxDir(self, vertical):
    return self.direction[vertical]
  def connect(self, obj, evt, fun):
    return obj.clicked.connect(fun)
  def setMode(self, mode = None, data = None):
    if self.Wid is None or self.Widget is None or mode is None: return
    if self.Wid == 'box' or self.Wid == 'groupbox':
      self.Widget.setDirection(self.boxDir(mode))
    elif self.Wid == 'listbox':
      if mode == 0: self.Widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
      elif mode == 1: self.Widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
      elif mode == 2: self.Widget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
      elif mode == 3: self.Widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
      elif mode == 4: self.Widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    if data is not None:
      self.setData(data)
  def show(self):
    if self.Wid.count("form"):
      self.FWidget.show()
    else:
      self.Widget.show()
  def close(self):
    if not(self.Wid.count("form")): return
    if self.Wid == "dlgform":
      self.Widget.done(0)
  def loop(self):
    self.Widget.exec_()
  def dlgFileOpen(self, titl = "Open File", path = "", filt = "*.*"):
    filename = QtWidgets.QFileDialog.getOpenFileName(self.Widget, self.Widget.tr(titl), path, self.Widget.tr(filt))
    if isinstance(filename, str):
      return filename
    if isinstance(filename, tuple) and len(filename) == 2:
      return filename[0]
    return filename
  def dlgFileSave(self, titl="Save File", path="", filt="*.*", confirmoverwrite=True):
    kwds = {} if confirmoverwrite else {'options': QtWidgets.QFileDialog.DontConfirmOverwrite}
    filename = QtWidgets.QFileDialog.getSaveFileName(self.Widget, 
                                                     self.Widget.tr(titl),
                                                     path, 
                                                     self.Widget.tr(filt),
                                                     **kwds)
    if isinstance(filename, str):
      return filename
    if isinstance(filename, tuple) and len(filename) == 2:
      return filename[0]
    return filename

class LBWidget: # A box containing a widget and label
  Wid = None
  widget = None
  Widget = None
  Label = None
  Box = None
  Parent = None
  isBox = False
  isLabel = False
  isWidget = False
  def __init__(self, parent = None, labeltext = None, boxmode = None, wid = None, data = None, opts = None):
    try:
      _ = LBWUSE
    except NameError:
      use()
    if LBWUSE == 'qt':
      self.LKeys = qtKeys()
      self.BKeys = qtKeys()
      self.WKeys = qtKeys()
    elif LBWUSE == 'wx':
      self.LKeys = wxKeys()
      self.BKeys = wxKeys()
      self.WKeys = wxKeys()
    self.setParent(parent)
    self.setup(labeltext, boxmode, wid, data, opts)
  def setParent(self, _parent = None):
    if _parent != None:
      self.Parent = _parent
      parent = _parent.Widget
      self.LKeys.setParent(parent)
      self.BKeys.setParent(parent)
      self.WKeys.setParent(parent)
  def setup(self, labeltext = None, boxmode = None, wid = None, data = None, opts = None):
    if wid != None:
      wid = wid.lower()
    widnull = False
    if wid == "label" and labeltext is None:
      labeltext = "" if data is None else data
      widnull = True
    if wid == "box" and labeltext is None:
      boxmode = 0
      widnull = true
    if widnull:
      wid = None
      data = None
      opts = None
    if labeltext != None and wid != None and boxmode is None:
      boxmode = 0
    self.setLabel(labeltext)
    self.setWidget(wid, data, opts)
    self.setBox(boxmode)
    if boxmode != None:
      self.isBox = True
      self.widget = self.Box
      if labeltext != None: self.BKeys.addWidget(self.Label)
      if wid != None: self.BKeys.addWidget(self.Widget)
    elif wid != None:
      self.isWidget = True
      self.widget = self.Widget
    elif labeltext != None:
      self.isLabel = True
      self.widget = self.Label
    if self.Widget is None:
      if self.Label != None:
        self.Widget = self.Label
        self.WKeys = self.LKeys
      elif self.Box != None:
        self.Widget = self.Box
        self.WKeys = self.BKeys
  def setLabel(self, labeltext = None):
    if labeltext is None: return
    self.Label, _ = self.LKeys.setWidget("label")
    self.LKeys.setData(labeltext)
  def setWidget(self, wid = None, data = None, opts = None):
    if wid is None: return
    self.Wid = wid.lower()
    if data != None and self.Wid in self.WKeys.nargin1:
      self.Widget, wdlef.FWidget = self.WKeys.setWidget(self.Wid, data)
    elif data != None and opts != None and self.Wid in self.WKeys.nargin2:
      self.Widget, self.FWidget = self.WKeys.setWidget(self.Wid, [data, opts])
    else:
      self.Widget, self.FWidget = self.WKeys.setWidget(self.Wid)
      self.setOpts(opts)
      self.setData(data)
  def setBox(self, _box = None):
    if _box is None: return
    if type(_box) is int:
      self.Box, _ = self.BKeys.setWidget("box")
      self.BKeys.setMode(_box)
    else:
      self.WKeys.setBox(_box.Widget)
  def add(self, child = None):
    if (child is None): return
    PKeys = self.BKeys if self.isBox else self.WKeys
    if PKeys is None: returns
    if child.isBox:
      PKeys.addBox(child.widget)
    elif child.isLabel:
      PKeys.addWidget(child.widget)
    else:
      if child.Label != None:
        PKeys.addWidget(child.Label)
      if child.Wid == 'buttongroup':
        PKeys.addWidget(child.WKeys.FWidget)
      else:
        PKeys.addWidget(child.Widget)
  def setChild(self, child):
    if child.Wid != 'childarea' and child.Wid != 'childform':
      self.WKeys.setBox(child.widget)
      return
    self.WKeys.setChild(child.widget)
  def addChild(self, label = ''):
    if self.Wid != 'childarea': return
    return self.WKeys.addChild(label)
  def setOpts(self, opts):
    if opts is None: return
    self.WKeys.setOpts(opts)
  def setData(self, data):
    if data is None: return
    self.WKeys.setData(data)
  def retData(self):
    return self.WKeys.retData()
  def retFrame(self):
    return self.WKeys.FWidget
  def valiData(self, hasentry = 0, isnumber = 0, d = None):
    if not(hasentry) and not(isnumber): return True, True
    if d is None: d = self.retData();
    if type(hasentry) is list and not(isnumber):
      for i in len(hasentry):
        if d == hasnentry[i]: return True, i
      return 0, None
    ld = len(d)
    if (isnumeric(d)):
      fd = float(d)
    else:
      fd = None
    valid0 = (ld > 0) if hasentry else 1
    valid1 = 0
    if type(isnumber) is list:
      if fd != None and len(isnumber) == 2:
        if isnumber[0] <= fd and fd <= isnumber[1]:
          valid1 = 1
    else:
      valid1 = isnumber
      if valid1 and fd is None: valid1 = 0
    valid = [valid0 != 0, valid1 != 0]
    return [valid0 != 0 and valid1 != 0, valid]
  def setMode(self, mode, data = None):
    self.WKeys.setMode(mode, data)
  def addSpace(self, space):
    self.WKeys.addSpace(space)
  def connect(self, sig, fun):
    self.WKeys.connect(self.Widget, sig, fun)
  def show(self):
    self.WKeys.show()
  def close(self):
    self.WKeys.close()
  def loop(self):
    self.WKeys.loop()
  def dlgFileOpen(self, titl = "Open File", path = "", filt = "*.*"):
    return str(self.WKeys.dlgFileOpen(titl, path, filt))
  def dlgFileSave(self, titl = "Save File", path = "", filt = "*.*", confirmoverwrite = True): # last ignored
    return str(self.WKeys.dlgFileSave(titl, path, filt, confirmoverwrite))

class BWidgets (LBWidget): # a box that contains one or more label-linked widgets
  Labels = []
  Wids = []
  Widgets = []
  nWids = 0
  Data = []
  Opts = []
  def __init__(self, parent = None, boxmodes = None, labellist = None, widlist = None, datalist = None, optslist = None):
    if boxmodes is None: boxmodes = [0, 0]
    boxmodes = repl(boxmodes, 2)
    LBWidget.__init__(self, parent, None, boxmodes[0])
    if widlist != None: self.setWidgets(parent, boxmodes[1], labellist, widlist, datalist, optslist)
  def setWidgets(self, parent = None, boxmode = None, labellist = None, widlist = None, datalist = None, optslist = None):
    if not(type(labellist) is list):
      nl = 1
      self.Labels = [labellist]
    else:
      nl = len(labellist)
      self.Labels = labellist
    if not(type(widlist) is list):
      nw = 1
      self.Wids = [widlist]
    else:
      nw = len(widlist)
      self.Wids = widlist
    self.n = max(nw, nl)
    if self.n > nw and nw == 1:
      self.Wids *= self.n
    elif self.n > nl and nl == 1:
      self.Labels *= self.n
    if datalist is None:
      self.Data = [None] * self.n
    elif type(datalist) is list:
      if type(datalist[0]) is list:  self.Data = datalist
      elif len(datalist) == self.n:  self.Data = datalist
      else: self.Data = [datalist] * self.n
    else:
      self.Data = [[datalist] * self.n]
    if optslist is None:
      self.Opts = [None] * self.n
    elif type(optslist) is list:
      if type(optslist[0]) is list: self.Opts = optslist
      elif len(optslist) == self.n: self.Opts = optslist
      else: self.Opts = [optslist] * self.n
    else:
      self.Opts = [[optslist] * self.n]
    self.Widgets = [None] * self.n
    for i in range(self.n):
      self.Widgets[i] = LBWidget(parent, self.Labels[i], boxmode, self.Wids[i], self.Data[i], self.Opts[i])
      self.add(self.Widgets[i])        # Adds label and Widget
  def retData(self):
    outData = []
    for lbw in self.Widgets:
      outData.append(lbw.retData())
    return outData
  def valiData(self, hasentry = 0, isnumber = 0):
    validata = []
    val = True
    for lbw in self.Widgets:
      v, discard = lbw.valiData(hasentry, isnumber)
      if not(v): val = False
      validata.append(v)
    return val, validata

class BGWidgets (LBWidget): # a button group that contains one or more labelled buttons
  def __init__(self, parent = None, boxlabel = None, boxmode = 0, labellist = [], wid = "radiobutton", checked = 0):
    widlist = [wid]*len(labellist)
    LBWidget.__init__(self, parent, None, None, 'buttongroup', boxlabel)
    if type(checked) is list:
      data = checked
    else:
      data = [False] * len(labellist)
      data[checked] = True
    self.Box = BWidgets(parent, [boxmode, boxmode], None, widlist, None, labellist)
    self.setChild(self.Box)
    if len(labellist):
      self.setWidgets(data)
  def setWidgets(self, data):
    for wid in self.Box.Widgets:
      self.WKeys.addWidget(wid.Widget)
    self.WKeys.setData(data)

class LBWidgets (LBWidget): # a labelled box that contains one or more label-linked widgets
  N = []
  Boxes = []
  def __init__(self, parent = None, boxlabel = None, boxmode = 0, labellist = None, widlist = None, datalist = None, optslist = None):
    LBWidget.__init__(self, parent, None, None, 'labelbox', boxlabel)
    self.Box = LBWidget(parent, None, 1)
    self.setChild(self.Box)
    if widlist != None:
      self.setWidgets(parent, boxmode, labellist, widlist, datalist, optslist)
  def setWidgets(self, parent = None, _boxmode = None, _labellist = None, _widlist = None, _datalist = None, _optslist = None):
    if not(type(_labellist) is list or type(_widlist) is list):
      labellist = [_labellist]
      widlist = [_widlist]
      if type (_datalist) is list: _datalist = [_datalist]
      if type (_optslist) is list: _optslist = [_optslist]
    else:
      [labellist, widlist] = rep2(_labellist, _widlist)
    self.N = len(widlist)
    if nDim(_datalist) < nDim(widlist):
      [datalist, _widlist] = rep2(_datalist, widlist)
    else:
      datalist = _datalist
    if len(datalist) != self.N:
      datalist = [datalist] * self.N
    if nDim(_optslist) < nDim(widlist):
      [optslist, _widlist] = rep2(_optslist, widlist)
    else:
      optslist = _optslist
    if len(optslist) != self.N:
      optslist = [optslist] * self.N
    self.Boxes = [None] * self.N
    for i in range(self.N):
      self.Boxes[i] = BWidgets(parent, [0, _boxmode], labellist[i],  widlist[i], datalist[i], optslist[i])
      self.Box.add(self.Boxes[i])
  def retData(self):
    outData = []
    for boxes in self.Boxes:
      outData.append(boxes.retData())
    return outData
  def valiData(self, hasentry = 0, isnumber = 0):
    validata = []
    val = True
    for boxes in self.Boxes:
      v, discard = boxes.valiData(hasentry, isnumber)
      if not(v): val = False
      validata.append(v)
    return val, validata
