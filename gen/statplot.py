import matplotlib.pyplot as mp
import numpy as np
import iplot
import fpfunc

class sumstatplot:
  def __init__(self, args = None):
    self.setplots(args)
    self.initialise()
  def initialise(self):  
    self.Label = []
    self.Value = []
    self.IsGeo = []
  def setplots(self, args = None):
    if args == None: return
    if not(type(args) is list):
      raise ValueError("Subplot specification must be a list of two subplot specifications")
    if len(args) != 2:
      raise ValueError("Subplot specification must be a list of two subplot specifications")
    args0 = args[0]
    args1 = args[1]
    self.itp = iplot.itxtplot(args0[0], args0[1], args0[2])
    self.isp = iplot.isubplot(args1[0], args1[1], args1[2])
  def addDataSet(self, label, value, isgeo = False):
    self.Label.append(label)
    self.Value.append(value)
    self.IsGeo.append(isgeo)
  def calcSumStats(self):
    n = len(self.Label)
    self.Means = np.empty(n, dtype = float)
    self.Stdvs = np.empty(n, dtype = float)
    for i in range(n):
      if self.IsGeo[i]:
        self.Means[i] = fpfunc.geomean(self.Value[i])
        self.Stdvs[i] = fpfunc.geostd(self.Value[i])
      else:
        self.Means[i] = np.mean(self.Value[i])
        self.Stdvs[i] = np.std(self.Value[i])
  def setTable(self):
    n = len(self.Label)
    self.Table = []
    for i in range(n):
      lbl = self.Label[i]      
      val = str(self.Means[i])
      pstr = r"$\propto$" if self.IsGeo[i] else r"$\pm$"
      err = ''.join([pstr, str(self.Stdvs[i])])
      self.Table.append([lbl, val, err])
  def constructTable(self):
    self.calcSumStats()
    self.setTable()
    self.itp.setData(self.Table, self.updateGraph)
  def updateGraph(self, event = None):
    i, j = self.itp.event2index(event)
    if j == None: return
    mp.sca(self.isp.aI)
    mp.cla()
    mp.ylabel(self.Label[j])
    y = self.Value[j]
    x = np.arange(len(y))
    mp.plot(x, y)    
    self.isp.reconnectGUI(self.isp.fI)
    mp.xlim([x[0], x[-1]])   
    mp.ylim([y.min(), y.max()])   
    self.isp.redrawFigure()    
    
      
  