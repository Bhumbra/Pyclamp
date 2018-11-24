from qmod import *

DEFIPDIR = ''
DEFOPDIR = '' 

class bqaexp(Qmodl):
  def openFile(self, spec = True):
    self.Base = lbw.LBWidget(None, None, None, 'base')
    self.pf = self.Base.dlgFileOpen("Open File", DEFIPDIR, "Tab-delimited (*.tdf *.tab *.tsv);;Excel file (*.xlsx *.xls);;All (*.*)")
    if self.pf is None: return
    rf =  self.readFile(self.pf, None, spec) 
    if rf is not None:
      self.SetNoise()
  def iniGUI(self):
    self.out = []
    self.iniForm()
    self.dock0 = pq.dock('')
    self.dock1 = pq.dock('')
    self.dock2 = pq.dock()
    self.area.add(self.dock0)
    self.area.add(self.dock1, 'right', self.dock0)
    self.area.add(self.dock2, 'bottom')
    self.gbox0 = self.dock0.addGbox()
    self.gbox1 = self.dock1.addGbox()
    self.bbox = self.dock2.addBbox()
    self.grap = pq.graph(parent=self.gbox0)
    self.plotMoments(self.grap)
    self.tabl = pq.tabl()
    self.tabl.setParent(self.gbox1)
    self.plotHatValues(self.tabl)
    self.bbox.addButton()
    self.bbox.setIconSize(0, QtCore.QSize(1,1))
    self.bbox.setText(0, 'Histograms')
    self.bbox.Connect(0, self.SetBins)
    if np.isnan(self.hatn):
      self.bbox.addButton()
      self.bbox.setIconSize(1, QtCore.QSize(1,1))
      self.bbox.setText(1, 'Export data')
      self.bbox.Connect(1, self.ExportData)
    else:
      self.bbox.addButton()
      self.bbox.setIconSize(1, QtCore.QSize(1,1))
      self.bbox.setText(1, 'Marginal 1D')
      self.bbox.Connect(1, self.PlotMarg1D)
      self.bbox.addButton()
      self.bbox.setIconSize(2, QtCore.QSize(1,1))
      self.bbox.setText(2, 'Marginal 2D')
      self.bbox.Connect(2, self.PlotMarg2D)
      self.bbox.addButton()
      self.bbox.setIconSize(3, QtCore.QSize(1,1))
      self.bbox.setText(3, 'Save')
      self.bbox.Connect(3, self.Archive)
    self.area.resize(self.defxy[0], self.defxy[1])
    self.form.resize(self.defxy[0], self.defxy[1])
    if self.stem is not None:
      try:
        wid = self.form.Widget
      except:
        wid = self.form
      wid.setWindowTitle(self.stem)
    return self.form
  def ExportData(self, ev = None):
    self.imv = [[]] * self.NX
    for i in range(self.NX):
      self.imv[i] = "".join( (self.labels[i], ": Mean=", str(self.mn[i]), "; Var.=", str(self.vr[i])) )
    self.Dlg = lbw.LBWidget(None, None, None, 'dlgform', self.pf)
    self.Box = lbw.LBWidget(self.Dlg, None, 1)
    self.LBi = lbw.LBWidget(self.Dlg, "Data selection", 1,"listbox", None, self.imv)
    self.LBi.setMode(3, range(self.NX))
    self.EDe = lbw.LBWidget(self.Dlg, 'Set Noise: ', 1, 'edit', str(self.e))
    self.BBx = lbw.BWidgets(self.Dlg, 0, None, ["Button", "Button", "Button"], ["Help", "Cancel", "OK"]) 
    self.BBx.Widgets[0].connect("btndown", self.SetResHL)
    self.BBx.Widgets[1].connect("btndown", self.SetResCC)
    self.BBx.Widgets[2].connect("btndown", self.ExportOK)
    self.Box.add(self.LBi)
    self.Box.add(self.EDe)
    self.Box.add(self.BBx)
    self.Dlg.setChild(self.Box)
    self.Dlg.show()      
  def SetResHL(self, ev = None):
    webbrowser.open_new_tab(MANDIR+MANPREF+MANHELP[2]+MANSUFF)
  def SetResCC(self, ev = None):
    self.Dlg.close()
  def ExportOK(self, ev = None):
    uimv = self.LBi.retData()
    OK = True
    OK = OK and len(uimv)
    OK = OK and self.EDe.valiData(1, 1)
    if not(OK): return
    _e = abs(float(self.EDe.retData()))
    self.Dlg.close()
    del self.Dlg
    _X = [[]] * len(uimv)
    _L = [[]] * len(uimv)
    for i in range(len(uimv)):
      _X[i] = nanravel(self.X[uimv[i]])
      _L[i] = self.labels[uimv[i]]
    self.ExportDlg(_X, _L, _e)
  def ExportDlg(self, _X = None, _L = None, _e = None):
    if _X is None: _X = self.X
    self.Base = lbw.LBWidget(None, None, None, 'base')
    if self.path is None or self.stem is None:
      defpf = "results.tsv" if _e is None else "results_e=" + str(e) + ".tsv"
      self.pf = self.Base.dlgFileSave("Export File", DEFOPDIR+"results.tsv", "Data file (*.tsv);;All (*.*)")
    else:
      opdn, opfs = self.path, self.stem
      _opdn = opdn.replace('analyses', 'results')
      if os.path.exists(_opdn): opdn = _opdn
      if len(DEFOPDIR):
        defop = DEFOPDIR + '/' + opfs + ".tsv" if _e is None else DEFOPDIR + '/' + opfs + "_e="+str(_e)+".tsv" 
      else:
        defop = opdn + '/' + opfs + ".tsv" if _e is None else opdn + '/' + opfs + "_e="+str(_e)+".tsv" 
      self.pf = self.Base.dlgFileSave("Export File", defop, "Data file (*.tsv);;All (*.*)")
    if self.pf is None: return
    if not(len(self.pf)): return
    writeQfile(self.pf, _X, _L)

import numpy as np
import pyqtgraph as pg
import lbwgui as lbw
import sys

def main():
  App = lbw.LBWidget(None, None, None, 'app')
  self = bqaexp()
  self.openFile()
  pg.QtGui.QApplication.exec_()

