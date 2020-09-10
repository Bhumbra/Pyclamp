from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.optfunc import *
from pyclamp.dsp.nnmod import *
from pyclamp.gui.pgb import *
import numpy as np

SHOWFITS = False

if SHOWFITS: 
  from mplqt import *
  import iplot
  import resfunc

# Numerical analysis
  
class MMA (censca): # Minimum / maximum analysis
  numlbl = 4
  def __init__(self, _centrescale = -1, _X = None):
    self.setCenSca(_centrescale)
    self.setLabel()
    self.analyse(_X)
  def setLabel(self):
    lbl = [[]] * self.numlbl
    lbl[0] = 'Maximum'
    lbl[1] = 'Minimum'
    lbl[2] = 'Maxind'
    lbl[3] = 'Minind'
    self.lbl = lbl
  def analyse(self, _X = None): 
    if _X is None: return
    self.Y = self.process(_X)
    self.N = self.Y.shape[0]
    self.n = self.Y.shape[1]
    self.maxi = self.Y.argmax(axis = 1)
    self.mini = self.Y.argmin(axis = 1)
    self.Z = np.empty((self.N, 4), dtype = float)
    self.Z[:,0] = self.Y[np.arange(self.N), self.maxi]
    self.Z[:,1] = self.Y[np.arange(self.N), self.mini]
    self.Z[:,2] = self.maxi
    self.Z[:,3] = self.mini

class DDD (censca): # deflection, derivative, deviation
  defdersep = 2
  numlbl = 7
  def __init__(self, _centrescale = -1, _X = None, _d = None):
    self.setCenSca(_centrescale)
    self.setLabel()
    self.analyse(_X, _d)
  def setLabel(self):
    lbl = [[]] * self.numlbl
    lbl[0] = 'Index'
    lbl[1] = 'Def+'
    lbl[2] = 'Def-'
    lbl[3] = 'Dif+'
    lbl[4] = 'Dif-'
    lbl[5] = 'Dev+'
    lbl[6] = 'Dev-'
    self.lbl = lbl
  def analyse(self, _X = None, d = None): 
    if _X is None: return
    if d is None: d = self.defdersep
    self.Y = self.process(_X)
    self.m = gckmode(nanravel(self.Y))
    self.M = self.Y - self.m
    MP = np.copy(self.M)
    MN = np.copy(self.M)
    MP[MP<0] = 0.
    MN[MP>0] = 0.
    self.N = self.Y.shape[0]
    self.n = self.Y.shape[1]
    self.D = diff0(self.Y, d)
    self.Z = np.empty((self.N, 7), dtype = float)
    self.Z[:,0] = np.arange(self.N, dtype = float)
    self.Z[:,1] = self.Y.max(axis = 1) - self.m
    self.Z[:,2] = self.Y.min(axis = 1) - self.m
    self.Z[:,3] = self.D.max(axis = 1)
    self.Z[:,4] = self.D.min(axis = 1)    
    self.Z[:,5] = np.mean(MP, axis = 1)
    self.Z[:,6] = np.mean(MN, axis = 1)
    
class EEE (MMA): # single/double exponential decay fits accommodating both polarities
  defPCT = [5, 20, 80, 98]
  numlbl = 24
  def __init__(self, _centrescale = -1, _X = None, _si = 1.):
    self.setCenSca(_centrescale)
    self.setLabel()
    self.setPCT(self.defPCT)
    self.analyse(_X, _si)
  def setPCT(self, _PCT = None):
    if _PCT is not None: self.PCT = np.array(_PCT, dtype = float)    
    self.FCT = self.PCT * 0.01;
  def setLabel(self):
    lbl = [[]] * (self.numlbl)
    lbl[0] = 'Maximum'
    lbl[1] = 'Minimum' 
    lbl[2] = 'Maxtime'
    lbl[3] = 'Mintime' 
    lbl[4] = 'Inivalue'
    lbl[5] = 'Initime'
    lbl[6] = 'IPR'  
    lbl[7] = 'IPT'
    lbl[8] = 'Asymptote'
    lbl[9] = 'Asymptime'
    lbl[10] = 'Exp_FitTime'
    lbl[11] = 'Exp_FitDur'
    lbl[12] = 'Exp_FitOffset'
    lbl[13] = 'Exp_FitAmp'
    lbl[14] = 'Exp_FitLDC'
    lbl[15] = 'Exp_FitMeanAD'
    lbl[16] = 'Exp_FitMaxAD'
    lbl[17] = 'Exp_Fit2Offset'
    lbl[18] = 'Exp_Fit2Amp0'
    lbl[19] = 'Exp_Fit2LDC0'
    lbl[20] = 'Exp_Fit2Amp1'
    lbl[21] = 'Exp_Fit2LDC1'
    lbl[22] = 'Exp_Fit2MeanAD'
    lbl[23] = 'Exp_Fit2MaxAD'
    self.lbl = lbl
  def analyse(self, _X = None, _si = 1., pgb = None):
    if _X is None: return
    if SHOWFITS: fig = iplot.figs()
    si = float(_si)
    MMA.analyse(self, _X)
    n2 = int(self.n/2)
    maxi, mini = np.array(self.Z[:,2], dtype = int), np.array(self.Z[:,3], dtype = int)
    posi = isconvex(self.Z, 1)
    self.Z[:,2:4] *= si
    self.Z = np.hstack((self.Z, np.empty((self.N, len(self.lbl)-self.Z.shape[1]), dtype = float)))
    if pgb is not None: pgb.init("Performing exponential decay analysis", self.N)
    for i in range(self.N):
      if pgb is not None: pgb.set(i)
      if posi[i]:
        pol = 1. 
        i0 = min(maxi[i], n2)
        Less = np.less_equal
      else:
        pol = -1.
        i0 = min(mini[i], n2)
        Less = np.greater_equal
      y0 = self.Y[i][i0:]
      f, F = indfrac(y0, self.FCT, -pol, True)
      M = np.median(y0[f[3]:])
      _m = np.nonzero(Less(y0[f[3]:], M))[0]
      m = f[0] + 5
      if len(_m): m = max(m, _m[0]+f[3])
      di = m - f[0]
      self.Z[i,4] = self.Y[i][i0]
      self.Z[i,5] = i0 * si
      self.Z[i,6] = F[2] - F[1]
      self.Z[i,7] = f[2] - f[1]
      self.Z[i,8] = M
      self.Z[i,9] = m * si
      self.Z[i,10] = float(f[0]) * si
      self.Z[i,11] = float(di) * si

      y = y0[f[0]:m]
      x = np.arange(len(y), dtype = float) * si
      hp = np.tile(np.nan, 3)
      Hp = np.tile(np.nan, 5)
      p0 = None
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          _hp = expdfit(x, y, 1)[0]
        hp = _hp
        p0 = np.copy(hp)
      except ValueError:
        print("expDecay Warning: exponential fit error")
      if p0 is not None:
        try:
          with  warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _Hp = expdfit(x, y, 2, p0)[0]
          Hp = _Hp
        except ValueError:
          print("expDecay Warning: exponential fit error")
      hy = expdval(hp, x)
      Hy = expdval(Hp, x)
      hd = np.minimum(np.fabs(hy - y), self.Z[i, 0]-self.Z[i, 1])
      Hd = np.minimum(np.fabs(Hy - y), self.Z[i, 0]-self.Z[i, 1])

      self.Z[i,12:15] = hp
      self.Z[i,15] = hd.mean()
      self.Z[i,16] = hd.max()
      self.Z[i,17:22] = Hp
      self.Z[i,22] = Hd.mean()
      self.Z[i,23] = Hd.max()
      if SHOWFITS:
        _t = si * (np.arange(len(self.Y[i])) - i0-f[0])
        fig.newPlot()
        plot(_t, self.Y[i], 'y')
        hold(True)
        plot(x, hy, 'r')
        plot(x, Hy, 'k')
        title(str(i))
    if pgb is not None: pgb.close()

class RFA (censca): # rise and fall analysis accommodating both polarities - note fall asymptote harmonised to max
  defPCT = [2, 20, 80] # (Fall asymptote fit percentile, IPR_0, IPR_1)
  lbl = None
  numlbl = 36
  def __init__(self, _centrescale = -1, _X = None, si = None):    
    self.setCenSca(_centrescale)
    self.setPCT(self.defPCT)
    self.setLabel()
    self.analyse(_X, si)    
  def setLabel(self):
    lbl = [[]] * self.numlbl
    lbl[0] = 'Dis'
    lbl[1] = 'Distime'
    lbl[2] = 'Max';
    lbl[3] = 'Min';
    lbl[4] = 'IPR(Rise)'
    lbl[5] = 'IPR(Fall)'
    lbl[6] = 'IPI(Rise)'
    lbl[7] = 'IPI(Fall)'
    lbl[8] = 'IPS(Rise)'
    lbl[9] = 'IPS(Fall)'
    lbl[10] = 'Asymptote(Rise)'
    lbl[11] = 'Asymptote(Fall)'
    lbl[12] = 'Asymptime(Rise)'
    lbl[13] = 'Asymptime(Fall)'    
    lbl[14] = 'FitDur(Rise)'
    lbl[15] = 'FitDur(Fall)'
    lbl[16] = 'FitOff(Rise)'
    lbl[17] = 'FitOff(Fall)'
    lbl[18] = 'FitAmp(Rise)'
    lbl[19] = 'FitAmp(Fall)'
    lbl[20] = 'FitLDC(Rise)'
    lbl[21] = 'FitLDC(Fall)'
    lbl[22] = 'FitMeanAD(Rise)'
    lbl[23] = 'FitMeanAD(Fall)'
    lbl[24] = 'FitMax(Rise)'
    lbl[25] = 'FitMax(Fall)'
    lbl[26] = 'Fit2Dur'
    lbl[27] = 'Fit2Amp'
    lbl[28] = 'Fit2Offset'
    lbl[29] = 'Fit2LAC'
    lbl[30] = 'Fit2LDC(Pos)'
    lbl[31] = 'Fit2LDC(Neg)'
    lbl[32] = 'Fit2LDC(Rise)'
    lbl[33] = 'Fit2LDC(Fall)'
    lbl[34] = 'Fit2MeanAD'
    lbl[35] = 'Fit2MaxAD'
    self.lbl = lbl[:]
  def setPCT(self, _PCT = None):
    if _PCT is not None: self.PCT = np.array(_PCT, dtype = float)    
    self.FCT = self.PCT * 0.01;
  def analyse(self, _X = None, _si = 1., _U = None, pgb = None):
    if _X is None: return
    if SHOWFITS: fig = iplot.figs()
    si = float(_si)
    self.maxldc = -np.log(si)
    self.Y = self.process(_X)
    self.N = self.Y.shape[0]
    self.n = self.Y.shape[1]
    n23 = int(float(self.n*2)/3.) # peak or trough must be in first 2/3
    if type(_U) is bool:
      self.U = np.tile(_U, self.N)
    elif isarray(_U):
      self.U = np.copy(_U)
    else:
      _U = isconvex(self.Y, 3)
      _U = np.sum(_U) >= 0.5*float(len(_U))
      self.U = np.tile(bool(_U), self.N) # overall vote
    self.Z = np.empty((self.N, len(self.lbl)), dtype = float)
    H = np.empty(self.N, dtype = int)
    rM = np.empty(self.N, dtype = int)
    fM = np.empty(self.N, dtype = int)
    rO = np.empty(self.N, dtype = int)
    fO = np.empty(self.N, dtype = int)
    for i in range(self.N):
      self.Z[i,0] = float(i)      # index      
      Yi = self.Y[i]
      maxi = np.argmax(Yi[:n23])
      mini = np.argmin(Yi[:n23])
      if self.U[i]:
        pol = -1.
        h = max(4, mini)
        Less = np.greater_equal
        fct = 1. - self.FCT
      else:
        pol = +1.
        h = max(4, maxi)
        Less = np.less_equal
        fct = self.FCT
      self.Z[i,0] = Yi[h]          # inflection
      self.Z[i,1] = float(h) * si # inflection time point (i.e. at maximum/maximum)
      self.Z[i,2] = Yi[maxi]       # maximum
      self.Z[i,3] = Yi[mini]       # minimum
      yr, yf = Yi[:h][::-1], Yi[h:]             # NOTE: the rising part is reversed
      _r, R = findfrac(yr, fct, [1, -pol], True)
      _f, F = findfrac(yf, fct, [1, -pol], True)
      r, f = np.array(np.round(_r), dtype = int), np.array(np.round(_f), dtype = int)
      _yr = yr[r[0]:]
      _yf = yf[f[0]:]
      Rm = np.median(_yr)
      Fm = np.median(_yf)
      _rm = np.nonzero(Less(_yr, Rm))[0]
      _fm = np.nonzero(Less(_yf, Fm))[0]
      rm, fm = 2, 2 # gives minimum separation (i.e. -2:2) for fit
      if len(_rm): rm = max(rm, _rm[0]+r[0])
      if len(_fm): fm = max(fm, _fm[0]+f[0])
      ro, fo = min(rm-3, r[2]), min(fm-3, f[2])
      H[i], rM[i], fM[i], rO[i], fO[i] = h, rm, fm, ro, fo
      self.Z[i,4] = R[2] - R[1] # inter-percentile range of rise 
      self.Z[i,5] = F[2] - F[1] # inter-percentile range of fall       
      self.Z[i,6] = float(_r[1]-_r[2])*si   # inter-percentile interval for rise
      self.Z[i,7] = float(_f[1]-_f[2])*si   # inter-percentile interval for fall
      self.Z[i,8] = float(-_r[1])*si     # inter-percentile start time for rise
      self.Z[i,9] = float(_f[1])*si      # inter-percentile stop time for fall        
      self.Z[i,10] = Rm  # Pre-rise asymptote value
      self.Z[i,11] = Fm  # Post-fall asymptote value
      self.Z[i,12] = float(-rm)*si # Pre-rise asymptote time
      self.Z[i,13] = float(fm)*si # Post-fall asymptote time
    fm = min(self.n-H.max(), fM.max())
    if pgb is not None: pgb.init("Fitting RFA profiles", self.N)
    for i in range(self.N):
      if pgb is not None: pgb.set(i)
      dz = self.Z[i, 2] - self.Z[i, 3] # maximum range across sweep

      # Extract inter-asymptotic range
      h = H[i]
      rm = rM[i]
      _i, i_ = h - rm, h + fm # inter-asymptotic range

      # Single fits
      r, f = h - rO[i], h + fO[i]
      yr = self.Y[i][_i:r]
      yf = self.Y[i][f:i_]
      xr = (np.arange(len(yr), dtype = float)+rO[i]) * si
      xf = (np.arange(len(yf), dtype = float)+fO[i]) * si
      pr = np.tile(np.nan, 3)
      pf = np.tile(np.nan, 3)
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          pr = expdfit(xr, yr, 1)[0]
      except ValueError:
        print("RFA Warning: rise fit error")
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          pf = expdfit(xf, yf, 1)[0]
      except ValueError:
        print("RFA Warning: fall fit error")
      hr = expdval(pr, xr)
      hf = expdval(pf, xf)
      dr = np.minimum(np.fabs(hr - yr), dz)
      df = np.minimum(np.fabs(hf - yf), dz)

      # Double fit
      pol = -1. if self.U[i] else +1.
      y = self.Y[i][_i:i_]
      x = np.arange(len(y), dtype = float) * si
      hp = np.tile(np.nan, 4)
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          hp = exp2fit(x, y, 1, pol)[0]
      except ValueError:
        print("RFA Warning: double fit error")
      hy = exp2val(hp, x)
      hd = np.minimum(np.fabs(hy - y), dz)
      ldcr, ldcf = max(hp[2], hp[3]), min(hp[2], hp[3])

      # Tabulate results of single fits

      self.Z[i,14] = float(len(xr)) * si
      self.Z[i,15] = float(len(xf)) * si
      self.Z[i,16] = pr[0]
      self.Z[i,17] = pf[0]
      self.Z[i,18] = pr[1]
      self.Z[i,19] = pf[1]
      self.Z[i,20] = pr[2]
      self.Z[i,21] = pf[2]
      self.Z[i,22] = dr.mean()
      self.Z[i,23] = df.mean()
      self.Z[i,24] = dr.max() if len(dr) else dz
      self.Z[i,25] = df.max() if len(df) else dz

      # Tabulate results of double fit

      self.Z[i,26] = float(len(x)) * si # Fit duration
      self.Z[i,27] = exp2amp(hp)
      self.Z[i,28:32] = hp
      self.Z[i,32] = ldcr
      self.Z[i,33] = ldcf
      self.Z[i,34] = hd.mean()
      self.Z[i,35] = hd.max() if len(hd) else dz
      if SHOWFITS and i < 500:
        _t = si * (np.arange(len(self.Y[i])) - h)
        fig.newPlot()
        plot(_t, self.Y[i], 'y')
        hold(True)
        tr = (np.arange(len(xr), dtype = float) + rO[i]) * si
        tf = (np.arange(len(xf), dtype = float) + fO[i]) * si
        plot(-tr[::-1], hr, 'r')
        plot(tf, hf, 'r')
        t2 = (np.arange(len(x), dtype = float) - rm) * si
        plot(t2, hy, 'b')
        title(str(i))
    if pgb is not None: pgb.close()

class FFT (censca): # fast fourier transform
  def __init__(self, _centrescale = -1, _X = None):    
    self.setCenSca(_centrescale)
    self.analyse(_X)
  def analyse(self, _X = None): 
    if _X is None: return
    self.Y = self.process(_X)
    self.Z = np.fft.fft(self.Y)

class DFT (FFT): 
  def __init__(self, _centrescale = -1, _X = None):    
    FFT.__init__(self, _centrescale, _X)
  def analyse(self, _X = None): 
    if _X is None: return
    self.Y = self.process(_X)
    self.z = np.fft.fft(self.Y)
    self.numlbl = self.z.shape[1]*2
    self.Z = np.empty((self.numlbl, self.z.shape[0]), dtype = float)
    self.lbl = [None] * self.numlbl
    j = 0
    for i in range(self.numlbl):
      if i % 2:
        self.Z[i,:] = np.imag(self.z[:,j])
        self.lbl[i] = "DFT_Imag_"+str(j)
        j += 1
      else:
        self.Z[i,:] = np.real(self.z[:,j])
        self.lbl[i] = "DFT_Real_"+str(j)
    self.Z = self.Z.T

    
class PCA (censca): # principal component analysis
  P = None  
  def __init__(self, _centrescale = -1, _uselog = False, _X = None):    
    self.setCenSca(_centrescale)
    self.setUseLog(_uselog)
    if _X is None: return
    self.analyse(_X)
  def setLabel(self):  
    self.lbl = [None] * self.n
    for i in range(self.n):
      self.lbl[i] = "Z_PC"+str(i+1)
  def setUseLog(self, _uselog = False):
    self.uselog = _uselog  
  def analyse(self, _X = None):
    if _X is None: return
    X = _X if not(self.uselog) else np.log(_X)
    self.Y = np.matrix(self.process(X))
    self.N = self.Y.shape[0]
    self.n = self.Y.shape[1]
    self.M = np.mean(self.Y, axis = 0)
    self.Y -= self.M
    [self.U, self.d, self.V] = np.linalg.svd(self.Y)    
    self.s = self.d / unzero(np.sqrt(float(self.N)-1.))
    self.e = np.power(self.s, 2)
    # Sign convention: largest element positive
    i = np.argmax(np.fabs(self.V), axis = 0) 
    I = i + np.arange(self.n) * self.n
    p = np.sign(np.ravel(self.V)[I]) 
    self.P = self.V.T
    self.P = np.multiply(self.P, np.tile(p.reshape([1, self.n]), (self.n, 1))) 
    self.Z = self.project()
    self.setLabel()
  def project(self, _X = None):
    if self.P is None: raise ValueError("PCA analysis must precede projection.") 
    if _X is None:
      X = self.Y
    else:
      X = _X if not(self.uselog) else np.log(_X)
      if self.cendim >= 0:
        X -= repl2dim(np.matrix(self.means), X.shape)
      if self.scadim >= 0:
        X /= repl2dim(np.matrix(self.stdvs), X.shape)  
      X = np.matrix(X, dtype = float)
      X -= self.M
    return X * self.P
  
class MWD (censca): # Mean windowed discursion (also outputs mean discursion)
  mmm = {False: np.max, True: np.min}
  argmmm = {False: np.argmax, True: np.argmin}
  defSth = 2.
  lbl = ['MD','MWD']
  def __init__(self, _centrescale = -1, _X = None, irange = None, _u = None):    
    self.setCenSca(_centrescale)
    self.setSth(self.defSth)
    self.analyse(_X, irange, _u)
  def setSth(self, _sth = None):
    if _sth is not None: self.sth = _sth
  def analyse(self, _X = None, irange = None, _u = None): 
    if _X is None: return
    self.Y = self.process(_X)
    [self.n, self.N] = self.Y.shape
    if irange is None:
      l2N = int(np.ceil(np.log2(float(self.N))))
      irange = [-l2N, l2N]
    self.di = irange # just to recall if necessary  
        
    # Try to remove excerpts by majority concensus vote on shape
    self.U = isconvex(self.Y, 3)
    if _u is None:
      _U = isconvex(self.Y, 3)
      self.u = np.sum(_U) >= 0.5*float(len(_U))
    else:
      self.u = _u
    self.uok = self.U == self.u                                  
    self.Yuok = self.Y[self.uok]
    if not(len(self.Yuok)):
      print("Warning from MWD.analyse(): No matched polarities - using entire data set.")
      self.Yok = self.Y
    self.muok = np.mean(self.Yuok, axis = 0)
    self.muoki = self.argmmm[self.u](self.muok) # Idealised central index point  
    
    # Isolate idealised window and calculate mean
    self.ij = np.array((self.muoki+irange[0], self.muoki+irange[1]), dtype = int)
    self.ij[0] = max(self.ij[0], 0)
    self.ij[1] = min(self.ij[1], self.N)
    self.Yij = self.Y[:,self.ij[0]:self.ij[1]]
    self.Z = np.hstack((np.mean(self.Y, axis = 1).reshape((self.n, 1)), np.mean(self.Yij, axis = 1).reshape((self.n, 1))))

class fpanal: 
  analkeys = {'DDD':0, 'RFA':1, 'DFT':2, 'PCA':3}
  analysis = [DDD, RFA, DFT, PCA]
  Z = None
  z = None
  Lbl = ''
  lbl = ''
  DDDd = 2
  def __init__(self, _X = None, _anal = None, _spec = None, _censca = -1):
    self.setAnal(_X, _anal, _spec, _censca)
  def setAnal(self, _X = None, _anal = None, _spec = None, _censca = -1):
    self.X = _X
    if self.X is None: return None
    self.anal = self.analkeys[_anal.upper()] if type(_anal) is str else _anal
    self.spec = _spec
    self.censca = _censca
  def analyse(self, si = None):
    self.Anal = self.analysis[self.anal](self.censca)
    if self.anal == 0:
      self.Anal.analyse(self.X, self.DDDd)
    elif self.anal == 1:
      self.Anal.analyse(self.X, si)
    else:
      self.Anal.analyse(self.X)
    self.Lbl = self.Anal.lbl
    self.lbl = self.Lbl[self.spec]
    self.Z = self.Anal.Z
    self.z = np.ravel(np.array(self.Z[:,  self.spec], dtype = float))
    return self.retRes()
  def retRes(self):
    return self.z
  def retLbl(self):
    return self.lbl

# Nominal analysis

class KNN (censca): # K-nearest neighbour
  defK = 5
  K = None  # K 
  VR = None # Voting rule
  def __init__(self, _centrescale = -1, _K = None, _VR = None):
    self.setCenSca(_centrescale)
    self.setKVR(_K, _VR)
  def setKVR(self, _K = None, _VR = None):
    if _K is None: _K = self.defK
    self.K, self.VR = _K, _VR
  def setRef(self, _Ref): # _Ref is a list of 2D arrays or matrices
    self.N = len(_Ref)
    self.n = np.empty(self.N, dtype = int)
    self.I = np.empty(0, dtype = int)
    for i in range(self.N):
      self.n[i] = len(_Ref[i])
      self.I = np.hstack( (self.I, np.tile(i, self.n[i])) )
      self.Ref = self.process(_Ref[i]) if not(i) else np.vstack((self.Ref, self.process(_Ref[i])))
    self.Ref = np.matrix(self.Ref, dtype = float)
    return self.analyse(self.Ref)
  def vote(self, _ind, _sed = None):
    k = self.K
    ind = _ind[:k]
    # Concensus trumps voting rules
    i = np.unique(_ind)
    if len(i) == 1: return i[0]
    # Only contests depend on voting rule
    vr = False if self.VR is None else self.VR
    b = np.zeros(self.N, dtype = bool)
    # Standard K-NN with k decrementation
    if not(vr):
      done = self.N < 2
      while not(done):
        c = np.empty(self.N, dtype = int)
        for i in range(self.N):
          c[i] = np.sum(ind[:k] == i)
        i = np.argsort(c)
        ci = c[i]
        if ci[-1] > ci[-2]:
          done = True
          i = i[-1]
          b[i] = True
        else:
          k -= 1
      return b
    # Minimum vote
    if Type(vr) is int:     
      for i in range(self.N):
        I = argtrue(ind == i)
        b[i] = len(I) >= vr
      return b
    sed = _sed[:k]
    # Maximum vote (vr = True) weighted 1./sed
    if Type(vr) is bool:
      w = 1./unzero(sed)
      w /= w.sum()
      W = np.zeros(self.N, dtype = float)
      for i in range(self.N):
        I = argtrue(ind == i)
        if len(I): W[i] = np.sum(w[I])
      i = np.argmax(W)
      b[i] = True
      return b
    # Minimum vote weighted -sed calibrating kth point to 0
    w = np.max(sed) - sed
    w /= w.sum()
    W = np.zeros(self.N, dtype = float)
    for i in range(self.N):
      I = argtrue(ind == i)
      if len(I): W[i] = np.sum(w[I])
    for i in range(self.N): 
      b[i] = W[i] >= vr
    return b
  def analyse(self, _X = None): 
    X = np.matrix(_X)
    if self.cendim >= 0:
      X -= repl2dim(np.matrix(self.means), X.shape)
    if self.scadim >= 0:
      X /= repl2dim(np.matrix(self.stdvs), X.shape)  
    X = np.matrix(X, dtype = float)
    self.SED = sqeucldis(X, self.Ref)
    m = len(self.SED)
    self.B = np.zeros( (m, self.N), dtype = bool)
    for i in range(m):
      sed = self.SED[i, :]
      srt = np.argsort(sed)
      self.B[i] = self.vote(self.I[srt], sed[srt])
    return self.B

