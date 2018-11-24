from pyplot import *
import pyqtplot as pq

class pywanim (pq.anim):
  roi = None
  roispec = None
  fps = None
  rcf = 1.
  ind = None
  Ind = None
  Act = None
  Anim = None
  _i = None
  def __init__(self, _obj = None):
    self.setObj(_obj)
    self.setRes()
  def setObj(self, obj = None):
    self.Obj = obj;
    if self.Obj is None: return
    if isinstance(self.Obj, pywave):
      self.Obj.pw = [self.Obj]
      self.Obj.parent = self.Obj.plot.gbox
      self.Obj.form = self.Obj.plot.parent()
    self.pw = self.Obj.pw
    self.npw = len(self.pw)
    if not(self.npw): return
    self.overlay = [_pw.overlay for _pw in self.pw]
    self.si = self.pw[0].si
    self.ns = self.pw[0].ns
    self.ne = self.pw[0].ne
    self.form = self.Obj.form
    self.parent = self.Obj.parent
  def setROI(self, _xx = None, _roispec = 0.5): # region of interest
    if _xx is None: _xx = 0
    if Type(_xx) is int: _xx = self.pw[_xx].xxyy[0]
    xx = list(_xx)
    if len(xx) == 1: xx = [xx[0], xx[0]]
    if Type(xx[0]) is float: xx[0] /= self.si
    if Type(xx[1]) is float: xx[1] /= self.si
    self.roi = [int(xx[0]), int(xx[1])]
    self.roispec = _roispec
    if self.fps is not None: self.calcInd()
  def setFPS(self, _fps = None, _rcf = 1.): # frames per sweep, range coefficient
    self.fps, self.rcf = _fps, _rcf
    if self.fps is None: self.fps = self.ns
    if self.roi is not None: self.calcInd()
  def calcInd(self, I = None, _xx = None):
    if I is None: I = self.ne
    if Type(I) is int: I = range(I)
    if _xx is None: 
      _xx = self.pw[0].xxyy[0] if self.roi is None else [0, self.roi[1] - self.roi[0]]
    xx = list(np.ravel(_xx))
    if len(xx) != 2: raise ValueError("Initial index range specification requres 2 values")
    if Type(xx[0]) is float: xx[0] /= self.si
    if Type(xx[1]) is float: xx[1] /= self.si
    m = int(0.5*float(xx[1] + xx[0]))
    d = int(0.5*float(xx[1] - xx[0]))
    self.active = self.pw[0].active
    self.onsets = np.ravel(self.pw[0].onsets)
    if self.rcf != 1.:
      M = np.tile(m, self.fps)
      c = np.hstack([1., self.rcf**np.arange(1, self.fps, dtype = float)])
      D = np.array(np.round(d*c), dtype = int)
    else:
      D = np.tile(d, self.fps)
      if self.roi is None:
        M = linspace(0., float(self.ns), self.fps)
      else:
        roim = np.mean(np.array(self.roi, dtype = float))
        rois = np.fabs(self.roi[1] - self.roi[0])*0.5
        if rois == 0.: rois = float(d)*0.5
        M = normspace(0, float(self.ns), self.fps, roim, self.roispec*rois)
        m = int(roim)
      M = np.array(np.round(M), dtype = int)
    self.M = M
    indMax = self.onsets[-1] + self.ns
    defactive = self.active.copy()
    defactive[2] = False
    maxf = len(I) * self.fps
    self.Ind = np.zeros((maxf, 2), dtype = int)
    self.act = np.zeros((maxf, 3, self.ne), dtype = bool) # for Concatenated view
    self.Act = np.zeros((maxf, 3, self.ne), dtype = bool) # for Overlay view
    self.Act[2] = False
    k = 0
    h = None
    for i in I:
      for j in range(self.fps):
        _m, _d = M[j], D[j]
        m_ = _m + self.onsets[i]
        ind = [m_ - _d, m_ + _d]
        if (ind[0] >= 0 and ind[1] < indMax) and ind[1] > ind[0]:
          self.Ind[k, :] = np.copy(ind)
          self.act[k, :] = defactive.copy()
          self.Act[k, :2, :i] = defactive[:2, :i]
          if _m >= m: 
            self.act[k, 2, i] = True
            self.Act[k, :2, i] = defactive[:2, i]
            self.Act[k, 2, i] = defactive[0, i]
            h = i
          elif h is not None:
            self.Act[k, 2, h] = defactive[0, i]
          k += 1
    self.Ind = self.Ind[:k, :]  
    return self.Ind
  def aniWave(self, i = None):
    if i is None:
      if self._i is None:
        self._i = -1
        for i in range(len(Ind)):
          self.aniWave(i)
        return
      else:
        i = self._i
    else:
      self._i = i - 1
    self._i = i + 1
    i = self._i
    for j in range(self.npw):
      if self.overlay[j]:
        self.pw[j].setactive(self.Act[i])
        self.pw[j].setLims([self.roi, self.pw[j].xxyy[1]])
      else:
        self.pw[j].setactive(self.act[i])
        self.pw[j].setLims([self.Ind[i], self.pw[j].xxyy[1]])
      self.pw[j].setWave()
  def animate(self, n = None, _patn = None, _pgb = None):
    if n is None: n =  len(self.Ind)
    self._i = -1
    self.initialise(self.pw[0].parent)
    self.setRes(self.width, self.height, self.fps)
    self.setAnimFunc(self.aniWave)
    return pq.anim.animate(self, n, _patn , _pgb)


    
