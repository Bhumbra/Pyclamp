# interactive wave display module

import matplotlib as mpl
import matplotlib.pyplot as mp
import numpy as np
import channel
from fpfunc import *
from fpanal import *
from lsfunc import *
from strfunc import *
from sifunc import *
from wffunc import *
import iplot as ip

LEFTBUTTON = 1
MIDDLEBUTTON = 2
RIGHTBUTTON = 3
DELETEKEY = 'D'
INSERTKEY = 'I'

class iwav (ip.isubplot): #  a generic class to display a wave matrix on a single subplot
  defselect = True
  defliving = True
  defembold = False
  pixmmm = int(2);  # pixel to maxmin multiple (must be even; 2 graphically optimal)
  zoomeps = 0.0001; # zoom delta detector epsilon for GUIs with concatentation
  augmy = 0.05      # Increase in y-axis (fractional w.r.t. range)
  showall = False   # Show all alive
  lcjet = False     # Flag to indicatejet colour scheme
  lcatt = 0.5       # Line colour attenuation of unselected 
  def __init__(self, _data = None, _chinfo = None,  _select = None, _living = None, _embold = None):
    self.initialise(_data, _chinfo, _select, _living, _embold)    
  def initialise(self, _data = None, _chinfo = None, _select = None, _living = None, _embold = None): 
    ip.isubplot.initialise(self)   
    self.initAxes()
    self.initLims()
    self.initCols()
    self.initLine()
    self.initData()
    self.overlay() # default to overlay
    self.setdata(_data, _chinfo, _select, _living, _embold)  
    self.setmarks()  
    self.clearPlot()
    self.mm = minmax()
    self.deleted = []
    self.keyPressFunc = []
    self.onResizeFunc = []
    self.onEmboldFunc = []
    self.onEmboldColFunc = None
    self.addOnEmboldFunc()
  def initAxes(self, autoX = False, autoY = False):
    if self.aI != None: ip.isubplot.disconnectGUI(self);
    if self.fI != None: self.disconnectGUI()
    self.fI = None
    self.aI = None
    self.setAutoXY(autoX, autoY) # this changes the interactivity of the GUI
    self.setLabels()
  def initLims(self):  
    self.axLims = np.empty(4) 
    self.tmin = 0
    self.tmax = 0
    self.i0 = 0
    self.i1 = 0
    self.t0no = None # a list for displayed t0 times -especially helpful when not overlaying
    self.tbeg = 0
    self.tend = 0
    self.nt = 0 # number of time points to display
    self.t = None
    self.w = None
    self.aw = None
    self.limitsSet = False
    self.bypassresize = False
  def initCols(self): 
    self.fgColour = None
    self.bgColour = None
  def initLine(self):
    self.lineColour = None  
    self.lineWidth = 0.5 
  def initData(self):
    self.samplint = 1
    self.data = None
    self.dataSet = False
    self.nActive = 0
    self.Ind = []
    self.ind = []
    self.nr = 0 # number of episodes
    self.nc = 0 # number of samples per episode
    self.N = 1
    self.n = [1]
    self.sumn = 1
    self.ns = 1
    self.setActive(self.defselect, self.defliving, self.defembold)
  def setdata(self, _data = None, _chinfo = None, _select = None, _living = None, _embold = None):
    if (_data == None or _chinfo == None):
      self.data = _data
      self.chinfo = _chinfo
      return     
    if _select == None: _select = self.defselect
    if _living == None: _living = self.defliving
    if _embold == None: _embold = self.defembold
    if self.data is not None and self.aI is not None:
      self.resetdata(_data, _chinfo, _select, _living, _embold)
      return
    if (type(_data) is list):
      self.data = _data
      self.N = len(self.data)
    elif _data.ndim > 1:
      self.data = [_data]
    else:
      self.data = [[_data]]   
    self.setChInfo(_chinfo)
    self.dt = self.samplint
    self.ns = nCols(self.data[0]) # number of samples per episode
    self.n = np.empty(self.N, dtype = int)
    self.waveID = [] # np.empty( (self.N, 2), dtype = int)
    self.nr = 0      # total number of episodes 
    self.minData = np.inf
    self.maxData = -np.inf
    for i in range(self.N):
      self.nc = nCols(self.data[i])
      if self.nc != self.ns:
        raise ValueError("Data input includes episodes of dissimilar sizes")
      self.n[i] = nRows(self.data[i])
      for j in range(self.n[i]):
        self.waveID.append(np.array( (i, j), dtype = int))
      self.nr += self.n[i]
      datai = self.data[i]
      if type(datai) is list:
        datai = datai[0]
      self.minData = self.gain * np.minimum(datai.min(), self.minData) + self.offset
      self.maxData = self.gain * np.maximum(datai.max(), self.maxData) + self.offset         
    randata = self.maxData - self.minData
    self.minData -= randata * self.augmy
    self.maxData += randata * self.augmy
    self.ranData = self.maxData - self.minData
    self.setEpoch()
    self.maxn = self.n.max()    
    self.sumn = self.n.sum()       
    self.dataSet = True      
    self.setActive(_select, _living, _embold)   # default to all data active  
    if self.name is None: self.name = 'Value'
    if not(len(self.name)): self.name = 'Value'  
    if self.units is None: self.units = 'units'
    if not(len(self.units)): self.units = 'units'  
    self.setLabels(None, "Time / s ", self.name + " / " + self.units) # default Labels    
  def resetdata(self, _data = None, _chinfo = None, _select = None, _living = None, _embold = None):
    if _select == None: _select = self.defselect
    if _living == None: _living = self.defliving
    if _embold == None: _embold = self.defembold
    if (type(_data) is list):
      self.data = _data
      self.N = len(self.data)
    elif _data.ndim > 1:
      self.data = [_data]
    else:
      self.data = [[_data]]   
    self.setChInfo(_chinfo)
    self.dt = self.samplint
    self.ns = nCols(self.data[0]) # number of samples per episode
    self.n = np.empty(self.N, dtype = int)
    self.waveID = [] # np.empty( (self.N, 2), dtype = int)
    self.nr = 0      # total number of episodes 
    self.minData = np.inf
    self.maxData = -np.inf
    for i in range(self.N):
      self.nc = nCols(self.data[i])
      if self.nc != self.ns:
        raise ValueError("Data input includes episodes of dissimilar sizes")
      self.n[i] = nRows(self.data[i])
      for j in range(self.n[i]):
        self.waveID.append(np.array( (i, j), dtype = int))
      self.nr += self.n[i]
      datai = self.data[i]
      self.minData = self.gain * np.minimum(datai.min(), self.minData) + self.offset
      self.maxData = self.gain * np.maximum(datai.max(), self.maxData) + self.offset         
    randata = self.maxData - self.minData
    self.minData -= randata * self.augmy
    self.maxData += randata * self.augmy
    self.ranData = self.maxData - self.minData
    self.setEpoch()
    self.maxn = self.n.max()    
    self.sumn = self.n.sum()       
    self.dataSet = True      
    self.setActive(_select, _living, _embold)   # default to all data active  
    
    # Reset limits
   
    self.initLims()
    self.clearPlot()
    self.setEpoch()
    self.setLimits()
    self.reconnectGUI()
    self.redraw(True)
  def setEpoch(self, tlo = None, thi = None):
    self.tbeg = tlo if tlo != None else 0.0;
    self.tend = thi if thi != None else float(self.ns) * self.samplint;
    self.Tend = self.tend * self.nr
  def setChInfo(self, _chinfo = None):    
    self.chinfo = channel.chWave()
    self.chinfo.index = 0
    self.chinfo.name = "Signal"
    self.chinfo.units = "units"   
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
          raise ValueError("iwave.iwav: channel info lists must contain numeric data only.")
      if len(_chinfo) > 1: 
        if type(_chinfo[0]) is int or type(_chinfo[0]) is float:
          self.chinfo.gain = _chinfo[1]
        else:
          raise ValueError("iwave.iwav: channel info lists must contain numeric data only.") 
      if len(_chinfo) > 2: 
        if type(_chinfo[0]) is int or type(_chinfo[0]) is float:
          self.chinfo.offset = _chinfo[2]
        else:
          raise ValueError("iwave.iwav: channel info lists must contain numeric data only.")
    elif isinstance(_chinfo, channel.chWave):
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
  def setmarks(self, _markt = None, _marks = None, _marky = 0.95):
    self.markt = _markt
    self.marks = _marks
    self.marky = _marky 
    if self.N > 1:
      raise ValueError("Cannot display marks for a >2D wave matrix")
    if _markt == None or _marks == None: return
    self.markt = np.matrix(_markt)
    self.marks = _marks if type(_marks) is list else [_marks]*self.markt.shape[0]
    if not(len(self.marks)): return
    if len(self.marks) != self.markt.shape[0]:
      raise ValueError("Marker time array size and marker length incommensurate")
  def addKeyPressFunc(self, _keyPressFunc = None):
    self.keyPressFunc.append(_keyPressFunc)  
  def addOnResizeFunc(self, _onResizeFunc = None):
    self.onResizeFunc.append(_onResizeFunc)  
  def addOnEmboldFunc(self, _onEmboldFunc = None, _onEmboldColFunc = None):
    if _onEmboldFunc is not None:
      self.onEmboldFunc.append(_onEmboldFunc) 
    if self.onEmboldColFunc is None and _onEmboldColFunc is None:
      self.onEmboldColFunc = self.emboldCol
    elif _onEmboldColFunc is not None:
      self.onEmboldColFunc = _onEmboldColFunc 
  def emboldCol(self, _RGB = None):
    dc = 0.25
    if _RGB is None: return _RGB
    if not(isarray(_RGB)): return _RGB
    if len(_RGB) != 3: return
    RGB = np.copy(_RGB) if type(_RGB) is np.ndarray else np.array(_RGB)
    i = 0
    sumc = RGB.sum()
    if sumc > 1.999:
      RGB = np.maximum(0.001, RGB - dc)
    else:  
      RGB = np.minimum(0.999, RGB + dc)
    return RGB
  def attenuate(self, _RGB = None): # attenuate colour for unselected
    if self.lcjet: return [0.5, 0.5, 0.5]
    if _RGB is None: return _RGB
    if not(isarray(_RGB)): return _RGB
    if len(_RGB) != 3: return
    RGB = np.copy(_RGB) if type(_RGB) is np.ndarray else np.array(_RGB)
    RGB *= self.lcatt
    return RGB
  def setAutoXY(self, autoX = None, autoY = None):
    if autoX != None and autoY != None:
      self.autoXY = np.array ( (autoX, autoY), dtype = int) 
    self.setGUI()  
  def setAxesColour(self, _fgColour = None, _bgColour = None):
    if _fgColour != None: self.fgColour = _fgColour        
    if _bgColour != None: self.bgColour = _bgColour
    if self.aI is None: return
    ip.setAxesColours(self.aI, self.fgColour, self.bgColour)
  def setLine(self, _lineColour = None, _lineWidth = None):
    if _lineColour != None: self.lineColour = _lineColour
    if _lineWidth != None: self.lineWidth = _lineWidth
    self.lcjet = False
    if _lineColour != 'jet': return
    self.lineColour = []
    for i in range(self.sumn):
      lcf = float(self.sumn - i) / float(self.sumn)
      self.lineColour.append(ip.rgbJet(lcf))
    self.lcjet = True
  def setActive(self, _select = None, _living = None, _embold = None):
    self.aw = None # this forces a screen recalculation
    if _select is not None: self.select = self.setStatus(_select)
    if _living is not None: self.living = self.setStatus(_living)
    if _embold is not None: self.embold = self.setStatus(_embold)
    self.setVisual()
  def setStatus(self, _status = True):
    if type(_status) is bool:
      status = [[]] * self.N      
      for i in range(self.N):
        status[i] = [_status] * self.n[i]    
      return status  
    elif type(_status) is list:
      return _status
    return [_status]
  def repStatus(self, _status):
    if type(_status) is bool:
      return self.setStatus(_status)
    else:
      status = [[]] * self.N      
      for i in range(self.N):
        status[i] = _status[i][:]
      return status  
  def setVisual(self):
    k = 0       
    self.nSelect = 0
    self.nEmbold = 0
    self.nLiving = 0     
    self.nActive = 0 
    self.nVisual = 0
    self.Ind = []
    self.ind = []
    self.active = [[]] * self.N    
    self.visual = [[]] * self.N
    self.visualID = [[]] * self.sumn
    self.isEmbold = [False] * self.sumn
    for i in range(self.N):
      self.active[i] = [False] * self.n[i]
      self.visual[i] = [False] * self.n[i]
      for j in range(self.n[i]):        
        if self.living[i][j]:
          self.nLiving += 1
          if self.select[i][j] or self.showall:
            self.nSelect += 1
            self.active[i][j] = True
            self.visual[i][j] = True                        
          if self.embold[i][j]:
            self.Ind.append(i)
            self.ind.append(j)
            self.nEmbold += 1
            self.visual[i][j] = True
          if self.active[i][j]:         
            self.nActive += 1
          if self.visual[i][j]:   
            self.isEmbold[self.nVisual] = self.embold[i][j]
            self.visualID[self.nVisual] = k
            self.nVisual += 1
        k += 1
    self.visualID = self.visualID[:self.nVisual]
    self.isEmbold = self.isEmbold[:self.nVisual]
  def setActiveIndexList(self, _selectIndexList = None): # only a list of lists will work (or with negative abbreviations)
    if _selectIndexList == None: self.setActive(False); return
    self.setActive(False, True, False)
    for i in range(len(_selectIndexList)):
      sil = _selectIndexList[i]
      if type(sil) is list:
        for j in range(len(sil)):
          if isnum(sil[j]):
            self.select[i][sil[j]] = True
      elif isnum(sil):
        if sil < 0:
          for j in range(-sil):
            self.select[i][j] = True
    self.setActive(self.select, True, False)    
  def retActiveIndexList(self):
    return bool2uint(self.active)
  def toggleEmbold(self, _i, _j):
    i = nanravel(np.array(_i, dtype = int))
    j = nanravel(np.array(_j, dtype = int))
    nij = len(i)
    for k in range(nij):
      self.embold[i[k]][j[k]] = not(self.embold[i[k]][j[k]])
    self.setActive() 
  def deleteEmbolded(self):
    if not(self.nEmbold): return
    deletelist = []
    for i in range(self.N):
      for j in range(self.n[i]):
        if self.embold[i][j]:
          self.living[i][j] = False
          self.embold[i][j] = False
          deletelist.append([i, j])
    self.deleted.append(deletelist[:])      
    self.setActive()       
  def insertLast(self):
    ndeleted = len(self.deleted) 
    if not(ndeleted): return
    ndeleted -= 1
    undeletelist = self.deleted[ndeleted]
    if self.nEmbold: self.embold = self.setStatus(False) # unembold all previous
    for undelete in undeletelist:
      i = undelete[0]
      j = undelete[1]
      self.living[i][j] = True
      self.embold[i][j] = True
    self.deleted = self.deleted[:ndeleted]    
    self.setActive()
  def retLimits(self):
    return [self.tmin, self.tmax, self.axLims[2], self.axLims[3]]
  def setLimits(self, _lims = None):
    if _lims is None:
      _lims = [-1, -1, None, None]
    self.setTimes(_lims[0], _lims[1])
    self.setYLims(_lims[2], _lims[3])
    self.limitsRet = self.retLimits()
    self.limitsSet = 0
  def setTimes(self, _tmin = -1, _tmax = -1):
    self.tmin = self.tbeg if (_tmin < self.tbeg or _tmin > self.tend) else _tmin;
    self.tmax = self.tend if (_tmax < self.tbeg or _tmax > self.tend) else _tmax; 
    self.i0 = int(round(self.tmin / self.samplint))
    self.i1 = int(round(self.tmax / self.samplint))  
    if self.i0 < 0: self.i0 = int(0) 
    if self.i1 >= self.ns : self.i1 = int(self.ns - 1)
    if self.overLay:
      self.setXLims(self.tmin, self.tmax)
      return
  # must recalculate x-limits for concatenated display
    self.Dt = self.tmax - self.tmin + self.dt
    self.setXLims(self.tmin, self.tmax + self.Dt*(self.nVisual - 1))
  def setXLims(self, _xmin = None, _xmax = None):
    tEnd = self.tend if self.overLay else self.Tend
    self.axLims[0] = _xmin if _xmin != None else self.tbeg
    self.axLims[1] = _xmax if _xmax != None else tEnd
  def setYLims(self, _ymin = None, _ymax = None):
    self.axLims[2] = _ymin if _ymin != None else self.minData
    self.axLims[3] = _ymax if _ymax != None else self.maxData  
  def singleepisode(self, episode = None, _tmin = -1, _tmax = -1):
    self.overLay = 1 # improves GUI performance
    er = 0
    ec = episode
    if type(episode) is list:
      if len(episode) == 2:
        er = episode[0]
        ec = episode[1]       
    self.setActive([0]) # zeros self.active    
    self.select[er][ec] = True
    self.setActive(self.select)
    self.setTimes(_tmin, _tmax);
  def toggleDisplay(self):
    self.overLay = not(self.overLay)
    if self.aI != None:
      tEnd = self.tend if self.overLay else self.Tend
      self.origxlim = (self.tbeg, tEnd)
      self.setTimes(self.tmin, self.tmax)
      self.drawPlot(True)   
  def concatenate(self, _select = None, _tmin = -1, _tmax = -1):
    self.overLay = 0
    self.setActive(_select)
    self.setTimes(_tmin, _tmax);
  def overlay(self, _select = None, _tmin = -1, _tmax = -1):
    self.overLay = 1
    self.setActive(_select)
    self.setTimes(_tmin, _tmax);  
  def setLabels(self, _title = None, _xlabel = None, _ylabel = None):
      self.tit = None
      self.xlb = None
      self.ylb = None
      self.title = _title
      self.xlabel = _xlabel
      self.ylabel = _ylabel
  def setGUI(self, _enableResize = True, _enableToggleDisplay = True, _enableDeletion = True, _emboldWeight = 8.):    # modify plot xyGUI to disable GUI if axes are automatically adjusted
    if (self.aI == None) : return
    ip.isubplot.setGUI(self, not(self.autoXY[0]), not(self.autoXY[1]), not(self.autoXY[0]), not(self.autoXY[1]))      
    #self.disconnectResizeEvent() # we handle resizing to here to refresh maxmin
    self.enableResize = _enableResize
    self.enableToggleDisplay = _enableToggleDisplay
    self.emboldWeight = _emboldWeight
    if (self.enableToggleDisplay or self.emboldWeigth > 0.): self.addStillClickFunc(self.on_still_click)
  def setGUILimits(self):    
    if (self.aI == None) : return
    if (self.limitsSet): # if GUI operational in axes, need to re-establish limits
      if not(self.autoXY[0]):
        lims = self.aI.get_xlim()
        t0 = lims[0]
        t1 = lims[1]
        if t1 < t0:
          t0 = lims[1]
          t1 = lims[0]             [self.Ind][self.ind]
        if not(self.overLay) and self.nVisual > 1: # if concatenating, may need to reintepret x-limits
          dt0 = self.tmax - self.tmin
          dt1 = (t1 - t0) / float(self.nVisual)
          if dt0 < self.zoomeps: dt0 = self.zoom.eps
          DT = dt1 / dt0
          if DT > (1.0 + self.zoomeps): # if user attempting to unzoom
            uz0 = abs(t0 - self.tmin)
            uz1 = abs(self.tmax - t1)
            uzt = uz0 + uz1
            if not(uzt): uzt = 1 
            DT = dt1 - dt0
            Dt0 = DT * uz0 / uzt
            Dt1 = DT * uz1 / uzt
            if t0: t0 = self.tmin - Dt0 # only adjust if not already on limit
            if t1 < self.tend: t1 = self.tmax + Dt1
            if t0 < self.tbeg: t0 = self.tbeg
            if t1 > self.tend: t1 = self.tend
          elif DT < (1.0 - self.zoomeps): # attempting to zoom in correcting times for episodes
            n0 = int( (t0-self.tmin) / self.Dt)
            DT = float(n0 * self.Dt)
            t0 -= DT        
            n1 = int( (t1-self.tmin) / self.Dt)
            if n1 > n0: # if user has crossed 2+ displays
              if t1 - DT - self.tmax > self.tmax - t0: # allow right limit to 'trump'
                t0 = self.tmin
                DT = float(n1 * self.Dt)
                t1 -= DT
              else:
                t1 = self.tmax
            else:
              t1 -= DT
          else: # user attempting to pan bizarrely on a concatenated display
            t1 = self.tmax + t0 - self.tmin    
        self.setTimes(t0, t1)
      if not(self.autoXY[1]):
        lims = self.aI.get_ylim()      
        self.setYLims(lims[0], lims[1]) 
    else:
      self.setTimes() 
      self.setYLims()        
  def plot(self, _figInst = None):
    return self.subplot(_figInst) 
  def subplot(self, argIn0 = None, argIn1 = None, argIn2 = None, argIn3 = None):
    if self.aI != None: 
      self.initAxes()
      self.initLims()
    if not(argIn3 is None):
      self.fI = argIn3
      mp.figure(self.fI.number)
    ip.isubplot.subplot(self, argIn0, argIn1, argIn2)    
    self.setGUI()       
    ip.setAxesColours(self.aI, self.fgColour, self.bgColour)
    if self.dataSet: self.redraw(True) #self.on_resize_event(None) # draw for the first time to ensure correct iplot limits    
    return self.aI       
  def drawMarks(self):
    if self.marks == None or self.markt == None: return
    if not(len(self.marks)): return
    t = self.markt[:, 1] * self.dt
    y = self.marky * (self.axLims[3] - self.axLims[2]) + self.axLims[2]
    for i in range(len(self.marks)):
      if self.markt[i,1] >= self.i0 and self.markt[i,1] <= self.i1 and self.active[0][self.markt[i,0]]:
        x = t[i,0]
        if not(self.overLay):
          x += self.t0no[self.markt[i, 0]] - self.tmin
        if self.fgColour is not None:  
          mp.text(x, y, self.marks[i], horizontalalignment='center',verticalalignment='center', color=self.fgColour)
        else:
          mp.text(x, y, self.marks[i], horizontalalignment='center',verticalalignment='center')
  def findPlot(self, _x, _y):
    I = None 
    J = None
    if self.overLay:
      x, y = int(np.floor(_x / self.dt)), float( (_y - self.offset) / self.gain)
      D = np.inf
      for i in range(self.N):
        for j in range (self.n[i]):
          if self.visual[i][j]:
            d = np.fabs(self.data[i][j][x] - y)
            if d < D:
              I = i
              J = j
              D = d
    else:
      k = np.floor( float(self.nVisual) * (_x  - self.axLims[0]) / (1e-300+(self.axLims[1] - self.axLims[0]) ) )
      h = 0
      for i in range(self.N):
        for j in range (self.n[i]):
          if self.visual[i][j]:
            if h == k:
              I = i
              J = j
            h += 1  
    return I, J            
  def validateRedraw(self): # do we really need to draw the plot  
    if self.aw is None:
      self.aw = ip.ret_axes_width(self.fI, self.aI)     
      if (not(self.overlay) and self.nVisual): # reduce axis width if concatenating
        self.aw /= self.nVisual
      self.aw = int(self.aw)         
      return self.setPrevDraw()
    else:      
      self.aw = ip.ret_axes_width(self.fI, self.aI)     
      if (not(self.overlay) and self.nVisual): # reduce axis width if concatenating
        self.aw /= self.nVisual
      self.aw = int(self.aw)  
    if self.prevaw != self.aw: return self.setPrevDraw()
    if self.prevoverLay != self.overLay: return self.setPrevDraw()
    if self.previ0 != self.i0: return self.setPrevDraw()
    if self.previ1 != self.i1: return self.setPrevDraw()  
    return not(self.overLay) # losing battle if concatenating     
  def setPrevDraw(self):
    self.prevaw = self.aw
    self.prevoverLay = self.overLay
    self.previ0 = self.i0
    self.previ1 = self.i1
    return True
  def clearPlot(self):
    self.pI = None
    self.bI = None
    if self.aI is not None: self.aI.cla()
  def calcPlot(self):     
    #if not(self.nVisual): return # there really is not point if there is nothing to show
    self.nt = self.mm.setup(self.aw * self.pixmmm, None, self.i0, self.i1)
    if self.t is not None: del(self.t) 
    if self.w is not None: del(self.w)       
    self.t = np.empty( (self.nVisual, self.nt) )
    self.w = np.empty( (self.nVisual, self.nt) )
    tlo = self.tmin
    thi = self.tmax
    self.t_0 = np.linspace(tlo, thi, self.nt)
    self.Dt = self.dt + thi - tlo;
    #if not(self.nt): return
    self.t0no = np.zeros(self.n[0])       
    k = -1
    for i in range(self.N):
      for j in range (self.n[i]):
        if self.visual[i][j]: 
          k += 1
          self.w[k] = self.mm.calc(self.data[i][j], self.i0, self.i1)
          if (self.gain != 1):
            self.w[k] *= self.gain
          if (self.offset != 0):  
            self.w[k] += self.offset
          if self.overLay:
            self.t[k] = self.t_0
          else:
            self.t[k] = np.linspace(tlo, thi, self.nt)                        
          if not(i): self.t0no[j] = tlo
          if not(self.overLay):
            tlo += self.Dt
            thi += self.Dt            
    if self.autoXY[0]:
      if not(self.overLay):
        thi -= self.Dt
        tlo = self.tmin
      self.setXLims(tlo, thi) 
    if self.autoXY[1]:    
      self.setYLims(self.w.min(), self.w.max()) 
  def drawLabels(self): # we also initialise the rubber-band format here
    if not(self.limitsSet):
      mp.figure(self.fI.number)
      mp.sca(self.aI)     
      if self.title != None: self.tit = mp.title(self.title)
      if self.xlabel != None: self.xlb = mp.xlabel(self.xlabel)
      if self.ylabel != None: self.ylb = mp.ylabel(self.ylabel)
    if self.fgColour is not None:       
      if self.tit is not None: self.tit.set_color(self.fgColour)
      if self.xlb is not None: self.xlb.set_color(self.fgColour)
      if self.ylb is not None: self.ylb.set_color(self.fgColour)
  def plotWave(self): # this plots the waves, marks, and bolds, but does not calculate
    if (self.fI == None or self.aI == None): return    
    mp.figure(self.fI.number)
    mp.sca(self.aI)  
    if not(self.overLay): self.update_axes()      
    if self.pI is None: self.initPlot()
    h = -1
    k = -1
    for i in range(self.N):
      for j in range (self.n[i]):
        h += 1
        if self.visual[i][j]:
          k += 1
          lc = self.lineColour
          if isarray(lc):
            if len(lc) == self.sumn:
              lc = lc[h]
          if not(self.select[i][j]):
            lc = self.attenuate(lc)
          pIh = self.pI[h]
          if type(pIh) is list: pIh = pIh[0]
          pIh.set_data(self.t[k], self.w[k])
          if lc is not None: pIh.set_color(lc)
        else:
          pIh = self.pI[h]
          if type(pIh) is list: pIh = pIh[0]
          pIh.set_data(None, None)
    self.aI.axis(self.axLims)
    self.drawMarks()  
    self.plotBold()            
  def initPlot(self):
    mp.figure(self.fI.number)
    mp.sca(self.aI)  
    self.pI = [[]] * self.sumn  
    for k in range(self.sumn):      
      self.pI[k] = mp.plot(None, None, '-', antialiased = False)
  def plotBold(self, animated = None):
    if animated is None: # option to bypass preunemboldening - give the impression of speed
      self.bgGUIbbox = self.initBold()
      animated = False
    elif not(animated):
      self.initBold(True) # treat it as animated as it is a recursion from animated call
    elif animated: 
      self.fI.canvas.restore_region(self.bgbbox)
    h = -1
    k = -1
    for i in range(self.N):
      for j in range (self.n[i]):
        h += 1
        nobold = True
        if self.visual[i][j]:
          k += 1
          if self.embold[i][j]:
            nobold = False
            lc = self.lineColour
            if isarray(lc):
              if len(lc) == self.sumn:
                lc = lc[h]
            if not(self.select[i][j]):
              lc = self.attenuate(lc)
            lc = self.onEmboldColFunc(lc)  
            lw = self.lineWidth[h] if isarray(self.lineWidth) else self.lineWidth        
            lw *= self.emboldWeight
            bIh = self.bI[h]
            if type(bIh) is list: bIh = bIh[0]
            bIh.set_data(self.t[k], self.w[k])
            if lc is not None: bIh.set_color(lc)
            if lw is not None: bIh.set_linewidth(lw)
        if nobold:
            bIh = self.bI[h]
            if type(bIh) is list: bIh = bIh[0]
            bIh.set_data(None, None)
            bIh.set_animated(False)
        if animated: 
          bIh.set_animated(True)
          self.aI.draw_artist(bIh)
    if animated: 
      self.fI.canvas.blit(self.aI.bbox)        
      self.plotBold(False)
    else:    
      self.draw()
  def initBold(self, animated = None):
    if animated is None: animated = False
    if self.bI is None:
      mp.figure(self.fI.number)
      mp.sca(self.aI)  
      self.bI = [None] * self.sumn  
    # Draw without embolding to capture canvas before-hand
    for h in range(self.sumn):
      if self.bI[h] is None: self.bI[h] = mp.plot(None, None, '-', antialiased = False)
      bIh = self.bI[h]
      if type(bIh) is list: bIh = bIh[0]
      if bIh.get_animated():
        bIh.set_data(None, None)
        self.aI.draw_artist(bIh)
        bIh.remove()
        self.bI[h] = mp.plot(None, None, '-', antialiased = False)
        bIh = self.bI[h]
        if type(bIh) is list: bIh = bIh[0]
      bIh.set_animated(False)  
      bIh.set_data(None, None)
    if animated:
      self.fI.canvas.restore_region(self.bgGUIbbox)
    self.fI.canvas.draw()
    return self.fI.canvas.copy_from_bbox(self.aI.bbox)
  def drawPlot(self, bypassLimitCheck = False, bypassMinMax = False, bypassValidation = False): # calculates
    if not(bypassLimitCheck): self.setGUILimits()  
    if not(bypassValidation) or self.aw is None: # Validation cannot be bypassed if we do not know the axis width
      if not(self.validateRedraw()): return
    if not(bypassMinMax): self.calcPlot()
    if self.origView: self.drawLabels()
    self.plotWave()    
    if not(self.limitsSet):    
      self.limitsSet = True       # newly established axes limits all done flag 
      #self.drawPlot(True, True)   # redraw with updated axes limits   
    if self.fgColour is not None and self.bgColour is not None: # update rubber-band colour to latest setting
      self.setRbFormat(list( (np.array(self.fgColour, dtype = float) + np.array(self.bgColour, dtype = float))/2.0))
  def on_resize_event(self, event = None):
    if self.bypassresize:
      self.bypassresize = False
      return
    self.drawPlot()
    if len(self.onResizeFunc):
      for onresizefunc in self.onResizeFunc:
        if onresizefunc is not None: onresizefunc(event)  
  def on_still_click(self, event): # this picks individual traces   
    if self.inMotion: return
    if event.button == MIDDLEBUTTON:
      self.toggleDisplay()
      if self.fI != None: self.drawPlot(); self.redraw()
    elif event.button == LEFTBUTTON: # embolden
      _nVisual = self.nVisual
      _Ind, _ind = self.findPlot(event.xdata, event.ydata)
      if _Ind is None or _ind is None: return
      if self.nEmbold > 1:
        self.embold = self.setStatus(False) # unembold all   
      elif self.nEmbold == 1:
        if _Ind != self.Ind[0] or _ind != self.ind[0]:
          self.embold = self.setStatus(False) # unembold all if not unembolding
      self.toggleEmbold(_Ind, _ind)
      # The next line unusually bypasses redraw-validation
      self.plotBold(True)
      # Perform external functions
      bypass = True
      if len(self.onEmboldFunc):
        for onemboldfunc in self.onEmboldFunc:
          if onemboldfunc is not None: 
            if onemboldfunc(event):
              bypass = False
      if not(bypass): self.drawPlot()        
  def on_key_press_event(self, event):
    if event.key.lower() == DELETEKEY.lower():
      self.deleteEmbolded() # flags a screen axis update
    elif event.key.lower() == INSERTKEY.lower():
      self.insertLast()
    if len(self.keyPressFunc):
      for keypressfunc in self.keyPressFunc:
        if keypressfunc is not None: keypressfunc(event)  
    self.drawPlot()

class iwavs (iwav): # display any number of wave episodes as a 2D matrix 
  defFigColour = np.array((0.1, 0.1, 0.1))
  defFgColour = np.array((1.0, 1.0, 1.0))
  defBgColour = np.array((0.0, 0.0, 0.0))
  lineColour = None   
  lineWidth = None
  axLims = np.empty(4)
  plotset = 0
  transpose = 0 # default view
  cidde = None
  GUIEnabled = 1
  transposing = 0
  def __init__(self, _data = None, _chinfo = None):    
    self.initialise(_data, _chinfo)    
    self.setDefColours()
    self.setTile()
  def setTile(self):
    for i in range(self.N):
      if i:
        self.Ni = np.concatenate( (self.Ni, np.tile(np.array(i, dtype = int), (1, self.n[i]))), 1)
        self.ni = np.concatenate( (self.ni, np.arange(0,self.n[i])), 1)
      else:
        self.Ni = np.tile(np.array(i, dtype = int), (1, self.n[i]))
        self.ni = np.arange(0,self.n[i])    
    self.wI = [] # iwav instantiation easier than inheritance with multiple axes
    for i in range(self.sumn): 
      waveInst = iwav(self.data[self.Ni[0,i]][self.ni[i]], [self.samplint, self.gain, self.offset])   
      self.wI.append(waveInst)      
  def setDefColours(self):
    self.setColours(self.defFigColour, self.defFgColour, self.defBgColour)
  def setColours(self, _figColour = None, _fgColour = None, _bgColour = None):
    if (_figColour != None): self.figColour = _figColour
    if (_fgColour != None): self.fgColour = _fgColour
    if (_bgColour != None): self.bgColour = _bgColour          
  def setPlots(self, _figInst = None, _transpose = None, _lineColour = None, _lineWidth = None):
    self.setSubPlots(_figInst, _transpose, _lineColour, _lineWidth)
  def setSubPlots(self, _figInst = None, _transpose = None, _lineColour = None, _lineWidth = None):
    if (_figInst == None): _figInst = mp.gcf()         
    if _lineColour != None: self.lineColour = _lineColour
    if _lineWidth != None: self.lineWidth = _lineWidth
    self.fI = _figInst
    self.fI.clf 
    self.fI.set_facecolor(self.figColour)
    if _transpose != None: self.transpose = not(self.transpose)  
    if self.transpose:
      self.nr = self.maxn
      self.nc = self.N    
      self.pn = self.Ni + self.N * self.ni
    else:
      self.nr = self.N
      self.nc = self.maxn
      self.pn = self.ni + self.maxn * self.Ni    
    self.aI = []
    for i in range(self.sumn):
      axInst = self.wI[i].subplot(self.nr, self.nc, self.pn[0,i]+1, self.fI)
      self.aI.append(axInst)    
      waveid = self.waveID[i] 
      self.wI[i].setAutoXY(0, 0)
      titl = ''.join( (str(waveid[0]), " (", str(waveid[1]), ")"))
      xlbl = ""
      ylbl = ""
      if not(i):
        ylbl = self.ylabel
      if i == self.sumn - 1:
        xlbl = self.xlabel
      self.wI[i].setLabels(titl, xlbl, ylbl)                
      self.wI[i].setLine(self.lineColour, self.lineWidth)
      self.wI[i].setYLims(self.axLims[2], self.axLims[3])      
      self.wI[i].setAxesColour(self.fgColour, self.bgColour)      
    self.plotset = 1           
    self.enableGUI()
    if _transpose == None: # first time subplots are called, we can enable transpose
      self.enableTranspose()     
  def enableGUI(self, enable = None):
    if not(self.plotset): return
    if enable != None: self.GUIEnabled = enable
    if self.GUIEnabled:
      for i in range(self.sumn):
        self.wI[i].connectGUI(self.fI)
    else:     
      for i in range(self.sumn):
        self.wI[i].disconnectGUI()
  def enableTranspose(self, enable = 1):
    if not(self.plotset): return
    if self.cidmd != None and enable == 1: return # no point reconnecting again
    if enable:
      self.cidmd = self.fI.canvas.mpl_connect("button_press_event", self.on_mouse_down)
      self.cidmu = self.fI.canvas.mpl_connect("button_release_event", self.on_mouse_up)
    else:
      self.fI.canvas.mpl_disconnect(self.cidmd)
      self.fI.canvas.mpl_disconnect(self.cidmu)
      self.cidmd = None      
      self.cidmu = None
  def aI2waveID(self, ai):
    for i in range(self.sumn):
      if ai == self.aI[i]:
        return self.waveID[i], i
    return None, None        
  def transposeTile(self):
    if self.transposing:
      self.setSubPlots(self.fI, 1) 
      for wi in self.wI:
        wi.drawPlot()
      self.fI.canvas.draw()  
    else:  
      for wi in self.wI:
        wi.initAxes()
        wi.initLims()
    self.transposing = not(self.transposing)  
  def on_mouse_down(self, event): 
    if (event.inaxes != self.aI): return  
    if event.inaxes == None:
      if event.button == MIDDLEBUTTON:  
        self.transposeTile()
  def on_mouse_up(self, event):
    if (event.inaxes != self.aI): return
    if self.transposing:
      self.transposeTile()
      
class iwavsel (iwav): # Wave episdoe selector using a single subplot     
  defFigColour = np.array((0.1, 0.1, 0.1))
  defFgOnColour = np.array((1.0, 1.0, 1.0))
  defFgOfColour = np.array((0.3, 0.3, 0.3))
  defBgOnColour = np.array((0.0, 0.0, 0.0))
  defBgOfColour = np.array((0.1, 0.1, 0.1))
  defLineWidth = 1
  lineColourOfAttenuation = 0.5
  def __init__(self, _data = None, _chinfo = None):
    self.initialise(_data, _chinfo)
  def initialise(self, _data = None, _chinfo = None):
    self.initAxes()
    self.initLims()
    self.setdata(_data, _chinfo)  
    self.setmarks()    
    self.setDefColours()    
  #self.setAutoXY(0, 0) # this defaults the GUI to be interactive
  def setdata(self, _data = None, _chinfo = None):
    iwav.setdata(self, _data, _chinfo)
    if _data is None: return
    self.setLines()
  def setLines(self, _lineColours = None, _lineWidths = None):
    self.lineColours = _lineColours
    self.lineWidths = _lineWidths        
    if not(self.dataSet): return    
    if _lineColours == None:
      self.lineColours = []
      for i in range(self.sumn):
        lcf = float(self.sumn - i) / float(self.sumn)
        self.lineColours.append(ip.rgbJet(lcf))
    else:
      self.lineColours = _lineColours
    if _lineWidths == None:
      self.lineWidths = np.tile( np.array( (self.defLineWidth) ), self.sumn)
    else:
      self.lineWidths = _lineWidths
    self.setLine(self.lineColours, self.lineWidths)  
  def setDefColours(self):
    self.setFigColour(self.defFigColour)
    self.setOnOffColours(self.defFgOnColour, self.defFgOfColour, self.defBgOnColour, self.defBgOfColour)
  def setFigColour(self, _figColour = None):
    if (_figColour != None): self.figColour = _figColour
  def setOnOffColours(self, _fgOnColour = None, _fgOfColour = None, _bgOnColour = None, _bgOfColour = None):
    if (_fgOnColour != None): self.fgOnColour = _fgOnColour
    if (_fgOfColour != None): self.fgOfColour = _fgOfColour
    if (_bgOnColour != None): self.bgOnColour = _bgOnColour
    if (_bgOfColour != None): self.bgOfColour = _bgOfColour
  def subplot(self, argIn0 = None, argIn1 = None, argIn2 = None, argIn3 = None):
    iwav.subplot(self, argIn0, argIn1, argIn2, argIn3)
    self.setAxesColour(self.fgOnColour, self.bgOnColour)
  #self.setLine(self.lineColours, self.lineWidths)

#self.wI[i].setAxesColour(self.bgOfColour, self.fgOfColour)
#self.wI[i].setLine(self.lineColours[i]*self.lineColourOfAttenuation, self.lineWidths[i])
#self.cidbpe = self.fI.canvas.mpl_connect("button_press_event", self.on_mouse_down)     
#self.cidbre = self.fI.canvas.mpl_connect("button_release_event", self.on_mouse_up)    
  #self.redraw()

class iwavsel (iwavs): # Wave episode selector based on 2D tile class
  defFgOnColour = np.array((1.0, 1.0, 1.0))
  defFgOfColour = np.array((0.3, 0.3, 0.3))
  defBgOnColour = np.array((0.0, 0.0, 0.0))
  defBgOfColour = np.array((0.1, 0.1, 0.1))
  defLineWidth = 1
  lineColourOfAttenuation = 0.5
  def __init__(self, _data = None, _chinfo = None):
    self.initialise()
    self.setdata(_data, _chinfo)
    self.setTile()
  def initialise(self):
    self.setDefColours()
    self.setAutoXY(0, 0) # this defaults the GUI to be interactive   
  def setLines(self, _lineColours = None, _lineWidths = None):
    if _lineColours == None:
      self.lineColours = []
      for i in range(self.sumn):
        lcf = float(self.sumn - i) / float(self.sumn)
        self.lineColours.append(ip.rgbJet(lcf))
    else:
      self.lineColours = _lineColours
    if _lineWidths == None:
      self.lineWidths = np.tile( np.array( (self.defLineWidth) ), self.sumn)
    else:
      self.lineWidths = _lineWidths
  def setDefColours(self):
    self.setFigColour(self.defFigColour)
    self.setOnOffColours(self.defFgOnColour, self.defFgOfColour, self.defBgOnColour, self.defBgOfColour)
  def setFigColour(self, _figColour = None):
    if (_figColour != None): self.figColour = _figColour
  def setOnOffColours(self, _fgOnColour = None, _fgOfColour = None, _bgOnColour = None, _bgOfColour = None):
    if (_fgOnColour != None): self.fgOnColour = _fgOnColour
    if (_fgOfColour != None): self.fgOfColour = _fgOfColour
    if (_bgOnColour != None): self.bgOnColour = _bgOnColour
    if (_bgOfColour != None): self.bgOfColour = _bgOfColour
  def setPlots(self, _figInst = None, _transpose = None, _lineColours = None, _lineWidths = None):
    self.setLines(_lineColours, _lineWidths)
    self.setSubPlots(_figInst, _transpose)
    for i in range(self.sumn):  
      self.setSubPlot(i) 
  # Draw then remove all existing GUI controls then replace with toggle callbacks 
    self.fI.canvas.draw()
    self.enableGUI(0) # remove all GUI controls
    self.cidbpe = self.fI.canvas.mpl_connect("button_press_event", self.on_mouse_down)     
    self.cidbre = self.fI.canvas.mpl_connect("button_release_event", self.on_mouse_up)      
  def setSubPlot(self, i):
    waveid = self.waveID[i]
    if self.active[waveid[0]][waveid[1]]:
      self.wI[i].setAxesColour(self.fgOnColour, self.bgOnColour)
      self.wI[i].setLine(self.lineColours[i], self.lineWidths[i])
    else:
      self.wI[i].setAxesColour(self.fgOfColour, self.bgOfColour)
      self.wI[i].setLine(self.lineColours[i]*self.lineColourOfAttenuation, self.lineWidths[i])  
  def toggleSelect(self, event):
    waveid, i = self.aI2waveID(event.inaxes)
    if waveid == None: return      
    self.select[waveid[0]][waveid[1]] = not(self.select[waveid[0]][waveid[1]])
    self.setActive()
    self.setSubPlot(i)
    self.wI[i].drawPlot()
    self.fI.canvas.draw() 
  def on_mouse_down(self, event):
    if event.inaxes == None:
      if event.button == MIDDLEBUTTON:  
        self.transposeTile()
    else:    
      self.toggleActive(event)
    
class imulwav: # A multichannel interactive wave displayer - multiple subplots of different data preclude direct inheritance from iwav
  defFigColour = np.array((0.1, 0.1, 0.1))
  defFgColour = np.array((1.0, 1.0, 1.0))
  defBgColour = np.array((0.0, 0.0, 0.0))
  lineColour = None   
  nCh = 0
  sumn = 0
  postFirstDraw = 0
  xLims = None
  wI = None
  Data = None
  def __init__(self, _Data = None, _ChInfo = None, _select = None, _living = None, _embold = None):
    self.initialise()
    self.setData(_Data, _ChInfo, _select, _living, _embold)
    self.setMarks()
  def initialise(self):
    self.setDefColours()
  def setData(self, _Data = None, _ChInfo = None, _select = None, _living = None, _embold = None):
    if (_Data == None or _ChInfo == None):
      return
    if (self.Data is not None and self.wI is not None):
      self.resetData(_Data, _ChInfo, _select, _living, _embold)
      return
    self.Data = _Data
    self.nCh = nLen(self.Data)
    self.ChInfo = repl(_ChInfo, self.nCh)    
    self.wI = []
    for i in range(self.nCh):
      wavInst = iwav(self.Data[i], self.ChInfo[i], _select, _living, _embold)
      self.wI.append(wavInst)
      if not(i):
        self.sumn = self.wI[i].sumn
      elif self.sumn != self.wI[i].sumn:
        print("Data sizes for channels incommensurate")
        raise
  def resetData(self, _Data, _ChInfo, _select = None, _living = None, _embold = None):
    self.postFirstDraw = 0
    self.Data = _Data
    self.nCh = nLen(self.Data)
    self.ChInfo = repl(_ChInfo, self.nCh)    
    for i in range(self.nCh):
      self.wI[i].setdata(self.Data[i], self.ChInfo[i], _select, _living, _embold)
      if not(i):
        self.sumn = self.wI[i].sumn
      elif self.sumn != self.wI[i].sumn:
        print("Data sizes for channels incommensurate")
        raise
    for i in range(self.nCh):
      self.wI[i].drawPlot(False, False, False)
  def setMarks(self, _Markt = None, _Marks = None):
    self.Markt = _Markt
    self.Marks = _Marks
    if _Markt == None or _Marks == None: return
    for i in range(len(self.wI)):
      self.wI[i].setmarks(self.Markt[i], self.Marks[i])
  def setActiveIndexList(self, _selectIndexList = None):
    for wi in self.wI:
      wi.setActiveIndexList(_selectIndexList)
  def retActiveIndexList(self):
    return self.wI[0].retActiveIndexList()
  def setDefColours(self):
    self.setColours(self.defFigColour, self.defFgColour, self.defBgColour)
  def setColours(self, _figColour = None, _fgColour = None, _bgColour = None):
    if (_figColour != None): self.figColour = _figColour
    if (_fgColour != None): self.fgColour = _fgColour
    if (_bgColour != None): self.bgColour = _bgColour     
  def setLines(self, _lineColour = None, _lineWidth = None):
    for wi in self.wI:
      wi.setLine(_lineColour, _lineWidth)    
  def setActive(self, _select = None, _living = None, _embold = None):
    for wi in self.wI:
      wi.setActive(_select, _living, _embold)    
  def setFigure(self, _figInst = None):
    if (_figInst == None): _figInst = mp.gcf()
    self.fI = _figInst
  def setSubPlots(self, _figInst = None, *args):
    self.setFigure(_figInst)
    Args = args if type(args) is tuple else ()
    if not(len(Args)):
      Args = []
      for i in range(self.nCh):
        Args.append([self.nCh, 1, i+1])
    self.fI.set_facecolor(self.figColour) 
    self.aI = []
    for i in range(self.nCh):      
      self.aI.append(self.wI[i].subplot(Args[i][0], Args[i][1], Args[i][2], self.fI))            
      self.wI[i].setAxesColour(self.fgColour, self.bgColour)
      self.wI[i].addOnResizeFunc(self.on_resize_event)
      self.wI[i].addOnEmboldFunc(self.onbold)
      # We have a bizarre situation when the delete/insert keys appear not to require coupling
  def drawSubPlot(self, n = None):
    if n != None:
      self.wI[n].drawPlot()
      return      
    for wi in self.wI:
      wi.drawPlot()   
  def setShowAll(self, _showall = False):
    for wi in self.wI:
      wi.showall = _showall
  def retLimits(self, _ch = None):
    if _ch == None: _ch = range(self.nCh)
    if type(_ch) is int:
      return self.wI[_ch].retLimits()
    else:
      _lims = []
      for i in range(len(_ch)):
        _lims.append(self.wI[_ch[i]].retLimits())
      return _lims
  def setLimits(self, _lims, _ch = None):
    if _ch == None: _ch = range(self.nCh)
    if type(_ch) is int:
      if length(_lims) == 1: _lims = _lims[0]
      self.wI[_ch].setLimits(_lims)
      self.wI[_ch].drawPlot(True)
    else:
      for i in range(len(_ch)):
        self.wI[_ch[i]].setLimits(_lims[i])
        self.wI[_ch[i]].drawPlot(True)
  def checkxLims(self):  
    if self.xLims is None:
      self.xLims = self.wI[0].aI.get_xlim()
      return True
    I = []
    _xLims = list(self.xLims)  
    for i in range(self.nCh):
      xLimsi = self.wI[i].aI.get_xlim()
      if not(isequal(self.xLims[0], xLimsi[0], 1)) or not(isequal(self.xLims[1], xLimsi[1], 1)):
        _xLims = xLimsi  
      else:
        I.append(i)
    if not(len(I)) or len(I) == self.nCh: return False
    self.xLims = _xLims
    for i in I:
      self.wI[i].setTimes(self.xLims[0], self.xLims[1])
      self.wI[i].bypassresize = True
      self.wI[i].drawPlot(True, False, True)
    return True 
    for i in range(self.nCh):    
      xlims = np.array( (self.aI[i].get_xlim()) )
      xrans = xlims[1] - xlims[0]
      xabsd = abs( np.array( (self.wI[i].axLims[0], self.wI[i].axLims[1]) ) - xlims )
      if xabsd.max() > xrans * d: redraw = int(i)
    if redraw > -1:
      self.wI[redraw].drawPlot()   
      tmin = self.wI[redraw].tmin
      tmax = self.wI[redraw].tmax
      notdrawn = np.ones(self.nCh, dtype = int)
      notdrawn[redraw] = int(0)
      for i in range(self.nCh):
        if notdrawn[i]:
          self.wI[i].setTimes(tmin, tmax)
          self.wI[i].drawPlot(True)
          notdrawn[i] = int(0)
    return redraw        
  def checkBold(self, event = None):
    if event is None: return
    I = []
    _i = None
    for i in range(self.nCh):
      if event.inaxes == self.wI[i].aI:
        _i = i
      else:
        I.append(i)
    if _i is None: return
    for i in I:
      self.wI[i].embold = self.wI[_i].repStatus(self.wI[_i].embold)
      _nVisual = self.wI[i].nVisual
      self.wI[i].setActive()
      # The next line unusually bypasses redraw-validation
      self.wI[i].bypassresize = True
      self.wI[i].drawPlot(True, _nVisual == self.wI[i].nVisual and _nVisual == self.wI[i].sumn, True) 
  def on_resize_event(self, event = None): # cannot overload since we do not use inheritance
    if self.postFirstDraw:      
      self.checkxLims()
    else:
      self.drawSubPlot()
      self.checkxLims()
      self.postFirstDraw = 1  
  def onbold(self, event = None):
    self.checkBold(event)  
    return True

class imulwavsel(imulwav): # as above but with a 2D selector for one channel (sCh)
  ws = None
  def __init__(self, _Data = None, _ChInfo = None, _selChan = 0):
    self.initialise()
    self.setData(_Data, _ChInfo)  
    self.setSelData(_selChan)
  def initialise(self):
    self.setDefColours()
  def setSelData(self, _selChan = 0):
    self.sCh = _selChan
    if self.ws == None:
      self.ws = iwavsel(self.Data[self.sCh], self.ChInfo[self.sCh])
    else:
      self.ws.setdata(self.Data[self.sCh], self.SamplInt[self.sCh], self.Gain[self.sCh], self.Offset[self.sCh])
  def setFigs(self, _figDis = None, _figSel = None):
    if (_figDis == None): _figDis = mp.gcf()
    if (_figSel == None): _figSel = mp.figure()
    self.fI = _figDis
    self.fIs = _figSel
    self.ws.setPlots(self.fIs)
    self.setLines(self.ws.lineColours)
    self.setPlots(self.fI)
    self.fI.canvas.mpl_disconnect(self.cidre)
    self.fIs.canvas.mpl_disconnect(self.ws.cidbpe)
    self.fIs.canvas.mpl_disconnect(self.ws.cidbre)
    self.cidre = self.fI.canvas.mpl_connect("resize_event", self.on_resize)
    self.cidbp = self.fIs.canvas.mpl_connect("button_press_event", self.on_mouse_down)   
    self.cidce = self.fIs.canvas.mpl_connect("resize_event", self.validateFigures) 
    self.fI.canvas.draw()
    self.fIs.canvas.draw()
  def validateFigures(self, event = None): # to close the other figure should one be closed
    fIExists = ip.figexists(self.fI)
    fIsExists = ip.figexists(self.fIs)
    if (fIExists and fIsExists): return      
    if (fIsExists):
      mp.close(self.fIs)
    else:
      mp.close(self.fI)    
  def setActiveIndexList(self, _activeIndexList = None):
    for wi in self.wI:
      wi.setActiveIndexList(_activeIndexList)
    self.ws.setActiveIndexList(_activeIndexList)  
  def retActiveIndexList(self):
    return self.ws.retActiveIndexList()      
  def on_resize(self, event):
    if self.postFirstDraw: 
      self.validateFigures()
      ch = self.validateXLims()
      self.fI.canvas.draw() # to update all subplots
      if ch > -1:
        wis = self.wI[self.sCh]
        tmin = wis.tmin
        tmax = wis.tmax
        ymin = wis.axLims[2]
        ymax = wis.axLims[3]
        for wi in self.ws.wI:
          wi.setTimes(tmin, tmax)
          if ch == self.sCh:
            wi.setYLims(ymin, ymax)
          wi.drawPlot(1)  
        self.fIs.canvas.draw() # to update all subplots      
    else:
      self.drawSubPlot()
      wis = self.wI[self.sCh]
      ymin = wis.axLims[2]
      ymax = wis.axLims[3]    
      for wi in self.ws.wI:
        wi.setYLims(ymin, ymax)
        wi.drawPlot(1)       
      self.postFirstDraw = 1 
  def on_mouse_down(self, event):
    self.ws.on_mouse_down(event)
    for wi in self.wI:
      wi.setActive(self.ws.active)
    self.drawSubPlot()  
    self.fI.canvas.draw() # to update all subplots

class iwavscat: # combines iscatter and iwav interactive though separate plots
  weol = None
  scat = None
  nscat = 0
  c = None  
  v = None
  x = None
  y = None    
  s = None  
  p = None
  m = None
  n = 0
  mc = True # monochrome
  plots = None
  xyplot = None
  defFigColour = np.array((0.1, 0.1, 0.1))
  defFgColour = np.array((1.0, 1.0, 1.0))
  defBgColour = np.array((0.0, 0.0, 0.0))
  lineColour = None     
  def __init__(self, _data = None, _chinfo = None, _x = None, _y = None, _c = None, _s = '.', _p = 'o', _m = 8):
    self.initialise(_data, _chinfo, _x, _y, _c, _s, _p, _m)
    self.setDefColours()
  def initialise(self, _data = None, _chinfo = None,  _x = None, _y = None, _c = None, _s = '.', _p = 'o', _m = 8):  
    if _data is None: return
    self.setData(_data, _chinfo, _x, _y)
    self.setmark(_c, _s)
    self.setpick(_p, _m)
  def setDefColours(self):
    self.setColours(self.defFigColour, self.defFgColour, self.defBgColour)
  def setColours(self, _figColour = None, _fgColour = None, _bgColour = None):
    if (_figColour != None): self.figColour = _figColour
    if (_fgColour != None): self.fgColour = _fgColour
    if (_bgColour != None): self.bgColour = _bgColour
  def setData(self, _data = None, _chinfo = None, _x = None, _y = None):
    if not(_data is None or _chinfo is None):
      if type(_data) is not np.ndarray:
        raise ValueError("Only 2D Numpy matrices are accepted as wave data")
      elif _data.ndim != 2:
        raise ValueError("Only 2D Numpy matrices are accepted as wave data")
      self.setweol(_data, _chinfo)        
      self.setxy(_x, _y)
    if not(self.n): return
    if _x is None or _y is None: return    
  def setweol(self, _data = None, _chinfo = None):
    self.weol = iwav(_data, _chinfo)
    self.n = self.weol.nr
    self.weol.addKeyPressFunc(self.onkey)
  def setxy(self, _x = None, _y = None):
    self.x, self.y = _x, _y
    if self.x is None or self.y is None: return
    if type(self.x) is not list: self.x = [self.x]
    if type(self.y) is not list: self.y = [self.y]
    self.nscat = len(self.x)
    if self.nscat != len(self.y):
      raise ValueError("X and Y inputs incommensurate")
    for i in range(self.nscat):
      self.x[i], self.y[i] = np.ravel(self.x[i]), np.ravel(self.y[i])
      xi, yi = self.x[i], self.y[i]
      if len(xi) != len(yi):
        raise ValueError("X and Y inputs incommensurate")
      if self.weol is not None:
        if len(xi) != self.n:
          raise ValueError("X/Y inputs incommesurate with wave data")    
  def setmark(self, _c = None, _s = '.'):
    self.c, self.s = _c, _s
  def setpick(self, _p = 'o', _m = 8):
    self.p, self.m = _p, _m
  def setFigure(self, _figInst = None):
    if (_figInst == None): _figInst = mp.gcf()
    self.fI = _figInst
  def setSubPlots(self, _figInst = None, *args):
    self.setFigure(_figInst)
    Args = list(args)
    if not(len(Args)):
      ns = float(self.nscat+1)
      nr = np.floor(np.sqrt(ns))
      nc = np.ceil(ns/nr)
      Args = [[int(nr), int(nc), 1]]
    while len(Args) <= self.nscat:
      Args.append(Args[-1][:])
      Args[-1][-1] += 1
    self.fI.set_facecolor(self.figColour) 
    self.plots =[[]] * (1+self.nscat)
    self.plots[0] = self.weol.subplot(Args[0][0], Args[0][1], Args[0][2], self.fI)
    ip.setAxesColours(self.plots[0], self.fgColour, self.bgColour)
    if self.c is None:
      self.weol.setLine('jet')
      self.c = self.weol.lineColour
    if self.v is None:
      if type(self.c) is list:
        self.v = [True] * len(self.c)
      else:
        self.v = True  
    self.weol.setAxesColour(self.fgColour, self.bgColour)
    self.weol.addOnEmboldFunc(self.onbold)
    self.mc = len(self.c) < self.n
    self.scat = [[]] * self.nscat
    self.xyplot = [[]] * self.nscat
    for i in range(self.nscat):  
      self.scat[i] = ip.iscatter(Args[i+1][0], Args[i+1][1], Args[i+1][2])
      self.plots[i+1] = self.scat[i].aI
      ip.setAxesColours(self.plots[i+1] , self.fgColour, self.bgColour)
      self.scat[i].setPicker(self.p, self.m, self.onpick)
      if self.mc:
        self.xyplot[i] = self.scat.plot(self.x[i], self.y[i], self.s)
      else:
        holdon = mp.ishold()
        self.xyplot[i] = [[]] * self.n
        for j in range(self.n):
          if not(j): 
            mp.hold('False')  
          self.xyplot[i][j] = self.scat[i].plot(self.x[i][j], self.y[i][j], self.s, mfc = self.c[j], mec = self.c[j])
          if not(i): 
            mp.hold('True')            
        mp.hold(holdon)
    return Args    
  def setMarkers(self, _c = None, _v = None):
    if _c is None:
      _c = self.c
    else:
      self.c = _c
    if _v is None:
      _v = self.v
    else:
      self.v = _v
    if type(self.v) is bool: self.v = [self.v] * self.n  
    for i in range(self.nscat):
      for j in range(self.n):
        self.xyplot[i][j].set_markerfacecolor(self.c[j])   
        self.xyplot[i][j].set_markeredgecolor(self.c[j])
        self.xyplot[i][j].set_visible(self.v[j])
  def onkey(self, event = None):
    if event.key.lower() == DELETEKEY.lower() or event.key.lower() == INSERTKEY.lower():
      self.setMarkers(self.c, self.weol.living[0])
      self.onbold(event)
      self.weol.draw() # this is needed to update all scatter canvases
  def onpick(self, event = None):
    _i = None
    _nVisual = self.weol.nVisual
    if event is not None: # must be an event
      Ind = 0
      ind = 0      
      for i in range(self.nscat):
        if event.inaxes == self.plots[i+1]:
          _i = i
      i = _i    
      ind = self.scat[i].ind if self.mc else self.scat[i].Ind
      self.weol.setActive(None, None, False) # this unembolds all at little cost
      if len(ind): self.weol.embold[Ind][ind] = True
      self.weol.setVisual() # this is necessary to update embold database
      self.onbold(_i) # this will update the other scatters      
      if self.weol.nVisual != _nVisual or self.weol.nVisual < self.n: # save recalculating if possible
        self.weol.drawPlot(True, False, True)
      else:
        self.weol.drawPlot(True, True, True)
    else: 
      self.onbold() # this will update all scatters
    return True     
  def onbold(self, event = None):
    Ind = 0
    ind = 0
    if self.mc:
      ind = self.weol.ind      
    else:
      Ind = self.weol.ind
    if type(event) is not int:  
      for i in range(self.nscat):
        self.scat[i].pick(Ind, ind)
    else:
      for i in range(self.nscat):
        if i != event:
          self.scat[i].pick(Ind, ind)
    self.fI.canvas.draw()
    return True
          
class iwavscatdisc (iwavscat): # combines iwavscat with a discriminating ellipse
  xyabpsr = []
  On = []
  on = []  
  onid = []
  showAll = True
  def __init__(self, _data = None, _chinfo = None, _x = None, _y = None, _c = None, _s = '.', _p = 'o', _m = 8):
    self.initialise(_data, _chinfo, _x, _y, _c, _s, _p, _m)
    self.setDefColours()
    self.setOnOffCols()
    self.setUseInside()
  def setOnOffCols(self, _onOffCols = [[0.8, 0.8, 0.8], [0.2, 0.2, 0.2]]):
    self.onOffCols = _onOffCols
  def setUseInside(self, _useInside = True):
    self.useInside = _useInside
  def setEllipse(self, ellstr = '', ell2str = 'o', _xyabps = None,  _ellipr = None, _ellipn = None, ellGUIthickness = [0.03, 0.04]):
    ellstr = repl(ellstr, self.nscat)
    ell2str = repl(ell2str, self.nscat)
    _xyabps = repl(_xyabps, self.nscat)
    _ellipr = repl(_ellipr, self.nscat)
    for i in range(self.nscat):
      self.scat[i].addGUIEllipse(self.on_ellipsemove, ellstr[i], ell2str[i], ellGUIthickness)
      self.scat[i].initGUIEllipse(_xyabps[i], _ellipr[i], _ellipn)
      self.scat[i].addStillClickFunc(self.toggleEllipse)
    self.useInside = repl(self.useInside, self.nscat)  
    self.On = np.empty((self.nscat, self.n), dtype = bool)
    self.on = np.empty(self.n, dtype = bool)    
    self.weol.addStillClickFunc(self.toggleShowAll)
    self.weol.setActive()
    self.weol.drawPlot() # this needs to be done to initialise the figure to default ellipse position(s)
    self.discriminate()
    self.showOnOff()
  def updateweol(self, bypasswavecalc = False):
    self.weol.setLine(self.c, 0.5)          
    self.weol.setActive([self.on], self.weol.living, self.weol.embold)  
    self.onid = np.nonzero(self.weol.active[0])[0]
    if self.showAll: self.weol.setActive(True, self.weol.living, self.weol.embold)
    if self.weol.nEmbold > 0: # update bold indices without redrawing
      self.onpick() # need to update colour of bolded events        
    self.weol.fI.canvas.set_window_title(str(len(np.nonzero(self.on)[0]))+"/"+str(len(self.on))+" events") 
    self.weol.drawPlot(self.weol.overLay, bypasswavecalc) # this redraws the entire figure        
  def discriminate(self):
    for i in range(self.nscat):
      self.On[i] = self.discriminateXY(self.x[i], self.y[i], self.scat[i].ell.xyabps, self.scat[i].ell.r, self.useInside[i])
      self.on = np.copy(self.On[i]) if not(i) else np.logical_and(self.on, self.On[i])
  def discriminateXY(self, _x, _y, _xyabps, _r, _useInside = True):
    discEllipse = ip.GUIEllipse(_xyabps, _r)
    if discEllipse.xyabps is None: return None
    a, r = discEllipse.calcNormel(_x, _y)
    if (_useInside):
      _on = r <= _r # the ball is on the line
    else:
      _on = r > _r
    return np.logical_and(_on, self.weol.living[0])  # eliminate killed at this stage
  def primeScat(self, event = None): # this primes other scatters to accommodate discrimination changes
    i = -1
    if event is not None:
      _i = -1
      for i in range(self.nscat):
        if event.inaxes == self.plots[i+1]:
          _i = i
      i = _i
    for _i in range(self.nscat):
      if i != _i:
        self.scat[_i].bgGUIbbox = None
        self.scat[_i].drawGUIPlots(True)
    return i 
  def showOnOff(self, bypasswavecalc = True):    
    for i in range(self.n):
      self.c[i] = self.onOffCols[0] if self.on[i] else self.onOffCols[1]       
    self.setMarkers(self.c)    
    self.updateweol(bypasswavecalc)    
  def toggleEllipse(self, event):
    if event.button == RIGHTBUTTON: 
      i = self.primeScat(event)
      if (i < 0): return      
      self.useInside[i] = not(self.useInside[i])
      self.setUseInside(self.useInside)
      self.discriminate()        
      self.showOnOff(self.showAll)
      self.primeScat()
  def toggleShowAll(self, event):
    if event.button == RIGHTBUTTON and event.inaxes == self.weol.aI:
      self.showAll = not(self.showAll)
      self.discriminate()        
      self.showOnOff(False)
  def on_ellipsemove(self, event = None, bypassprimescat = False):
    self.discriminate()
    bypasswavecalc = False if self.weol.nEmbold else self.showAll
    self.showOnOff(bypasswavecalc)
    if not(bypassprimescat): self.primeScat(event)
  def onkey(self, event = None): # we need this just to update the on-event count
    iwavscat.onkey(self, event)
    self.on_ellipsemove(event)
    
class iwavscatdiscbool (iwavscatdisc): # combines 3x iwavscatdisc with a Boolean colour scheme
  defColMode = 0 # 0 = monochrome, 1 = logical, 2 = ordinal
  defLogCols = [[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]]
  defUseBool = [ True,   True,   True,   True,   False,  False,  False,  False]
  defLogical = [[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
  offColMult = 0.1
  c8 = None
  ci = None
  logic3 = None
  buttonsSet = False
  def __init__(self, _data = None, _chinfo = None, _x = None, _y = None, _c = None, _s = '.', _p = 'o', _m = 8):
    iwavscatdisc.__init__(self, _data, _chinfo, _x, _y, _c, _s, _p, _m)
    self.setColMode(self.defColMode)
    self.setLogCols(self.defLogCols)
    self.setUseBool(self.defUseBool)
    self.setLogical(self.defLogical)
  def setColMode(self, _colMode = None):
    if _colMode is not None: self.colMode = _colMode  
  def setLogCols(self, _logCols = None):
    if _logCols is not None: self.logCols = _logCols
    if type(self.logCols is list):
      self.logCols = np.array(self.logCols, dtype = float)
    self.n8 = np.empty(8, dtype = int)
  def setUseBool(self, _useBool = None):
    if _useBool is not None: self.useBool = _useBool
  def setLogical(self, _logical = None):
    if _logical is not None: self.logical = _logical   
  def setweol(self, _data = None, _chinfo = None):
    self.showAll = False # more appropriate default
    self.weol = iwav(_data, _chinfo)
    self.n = self.weol.nr
    self.weol.addKeyPressFunc(self.onkey)
    self.weop = iwav(_data, _chinfo)
  def setSubPlots(self, _figInst = None, *args):
    if self.nscat != 3:
      raise ValueError("Three scatters mandatory")
    Args = list(args)
    if not(len(Args)):
      Args = [[2, 2, 4], [2, 3, 1], [2, 3, 2], [2, 3, 3], [2, 2, 3]]
    self.weop.setAxesColour(self.fgColour, self.bgColour)    
    iwavscatdisc.setSubPlots(self, _figInst, Args[0], Args[1], Args[2], Args[3])    
    self.plots.append(self.weop.subplot(Args[4][0], Args[4][1], Args[4][2], self.fI))
    self.weop.setGUI(True, True, False, 1.) # no point empowering bolding here
    self.weop.setLine(self.weol.lineColour)
    self.weop.addOnEmboldFunc(self.onbold)
    self.weop.drawPlot()
    self.setEllipses()
  def setEllipses(self, ellstr = ['r', 'g', 'b'], ell2str = ['ro', 'go', 'bo'], _xyabps = None,  _ellipr = None, _ellipn = None, ellGUIthickness = [0.03, 0.04]):
    self.setEllipse(ellstr, ell2str, _xyabps, _ellipr, _ellipn, ellGUIthickness)
  def setButtons(self):
    if not(self.buttonsSet):
      self.paxes = [[]] * 8
      self.taxes = [[]] * 8
      self.pbtns = [[]] * 8
      self.tbtns = [[]] * 8
      for i in range(8):
        self.paxes[i] = ip.exes('l', 7-i, 18)
        self.taxes[i] = ip.exes('r', 7-i, 18)
        self.pbtns[i] = mpl.widgets.Button(self.paxes[i], ' ')
        self.tbtns[i] = mpl.widgets.Button(self.taxes[i], ' ')
        self.pbtns[i].color = tuple(self.logCols[i])
        self.pbtns[i].hovercolor = tuple(self.logCols[i])
        self.pbtns[i].on_clicked(self.on_pbtn)
        self.tbtns[i].on_clicked(self.on_tbtn)
      self.buttonsSet = True
    for i in range(8):
      col = np.array(self.logCols[i], dtype = float)
      if not(self.useBool[i]): col *= self.offColMult
      self.tbtns[i].color = tuple(col)
      self.tbtns[i].hovercolor = tuple(col)                  
  def discriminate(self):
    for i in range(self.nscat):
      self.On[i] = self.discriminateXY(self.x[i], self.y[i], self.scat[i].ell.xyabps, self.scat[i].ell.r, self.useInside[i])
    ofon = [np.logical_not(self.On), self.On]
    if self.logic3 is None: self.logic3 = np.empty((3, self.n), dtype = bool)
    if self.c8 is None: self.c8 = np.empty((8, self.n), dtype = bool)
    if self.ci is None: self.ci = np.empty(self.n, dtype = int)
    self.on = np.zeros(self.n, dtype = bool)
    for i in range(8):
      for j in range(3):
        self.logic3[j] = ofon[self.logical[i][j]][j]
      self.c8[i] = np.logical_and(self.logic3[0], np.logical_and(self.logic3[1], self.logic3[2]))
      cind = np.nonzero(self.c8[i])[0]
      self.n8[i] = len(cind)
      self.ci[cind] = i
      if self.useBool[i]: self.on[cind] = True  
  def updateweop(self):
    self.weop.setLine(self.c, 0.5)
    #self.weop.setActive(self.weol.select, self.weol.living, self.weol.embold)    
    self.weop.setActive(self.weol.embold, self.weol.living, False)    
    self.weop.drawPlot(self.weop.overLay) # this redraws the entire figure        
  def emboldLogical(self, i):
    if not(self.n8[i]): return 
    i = self.ci == i
    self.weol.setActive(None, None, [list(i)])
    self.onbold()    
  def showOnOff(self, bypasswavecalc = False):    
    self.c = self.logCols[self.ci]      
    self.setButtons()
    self.setMarkers(self.c)
    self.updateweop()
    self.updateweol(bypasswavecalc)    
  def onkey(self, event):
    argout = iwavscatdisc.onkey(self, event)
    self.updateweop()
  def onpick(self, event = None):
    argout = iwavscatdisc.onpick(self, event) # this will call onbold
    return argout
  def onbold(self, event = None):
    argout = iwavscatdisc.onbold(self, event)
    self.updateweop()
    return argout
  def onkey(self, event = None):
    argout = iwavscatdisc.onkey(self, event) 
    self.updateweop()
    return argout 
  def on_pbtn(self, event = None):
    _i = None
    for i in range(8):
      if event.inaxes == self.paxes[i]: _i = i
    if _i is None: return 
    self.emboldLogical(_i)
    self.updateweop()
    self.weol.drawPlot(self.weol.overLay)
  def on_tbtn(self, event = None):
    _i = None
    for i in range(8):
      if event.inaxes == self.taxes[i]: _i = i
    if _i is None: return
    self.useBool[_i] = not(self.useBool[_i])
    self.on_ellipsemove(event, True)
 
class iwavscatdiscboolcalc (iwavscatdiscbool): # combines iwavscatdiscbool with calculation ability
  def __init__(self, _data = None, _chinfo = None, _c = None, _s = '.', _p = 'o', _m = 8):
    iwavscatdiscbool.__init__(self, _data, _chinfo, None, None, _c, _s, _p, _m)
    self.setColMode(self.defColMode)
    self.setLogCols(self.defLogCols)
    self.setUseBool(self.defUseBool)
    self.setLogical(self.defLogical)
    if _data is not None:
      self.calcXY()
  def calcXY(self):
    if self.weol.data is None: return
    X = self.weol.data
    if type(X) is list: 
      if len(X) > 1:
        raise ValueError("Input data must be an array datatype")
      else:
        X = X[0]
    self.ddd = DDD(X, 2, [-1, 1])
    self.rfa = RFA(X, self.weol.dt, [-1, -1])
    self.fft = FFT(X, [-1, 1])
    self.pca = PCA(X, [-1, 1])
    self.xxx = [self.rfa.Z[:, 3], real(self.fft.Z[:,1]), self.pca.Z[:,0]] 
    self.yyy = [self.rfa.Z[:,10], imag(self.fft.Z[:,1]), self.pca.Z[:,1]]
    self.setxy(self.xxx, self.yyy) 

