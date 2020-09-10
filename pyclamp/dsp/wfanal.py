import sys
import os
import warnings
CWDIR = os.getcwd()
GENDIR = CWDIR + '/../gen/'
sys.path.append(GENDIR)
import numpy as np
import scipy.stats as stats
from pyclamp.dsp.wffunc import *
from pyclamp.dsp.wfprot import *
from pyclamp.dsp.fpfunc import *
from pyclamp.dsp.fpanal import *

if SHOWFITS: import matplotlib.pyplot as mp

# Note that centre-scale facilities are disabled here

class spikeShape(MMA):
  defPCT = [20, 80, 98]
  numlbl = 28
  defdersep = 2
  ptht = [-1.8, 0.2] # peri-threshold detection time w.r.t. to ascent-max time (e.g. use [-1.8, 0.2] for htan fit)
  pthc = -0.5*np.pi # threshold critical value (e.g. use -0.5*np.pi for htan fit)
  def __init__(self, _X = None, _si = 1.):
    self.setCenSca()
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
    lbl[4] = 'Ascent'
    lbl[5] = 'Descent'
    lbl[6] = 'Ascenttime'  
    lbl[7] = 'Descenttime' 
    lbl[8] = 'Ascentvalue'
    lbl[9] = 'Descentvalue'
    lbl[10] = 'Threshold'
    lbl[11] = 'Threshtime'
    lbl[12] = 'FWHM'
    lbl[13] = 'FullWidth'
    lbl[14] = 'AHP_IPR'
    lbl[15] = 'AHP_IPT'
    lbl[16] = 'AHP_Asymptote'
    lbl[17] = 'AHP_Asymptime'
    lbl[18] = 'AHP_FitDur'
    lbl[19] = 'AHP_FitOffset' 
    lbl[20] = 'AHP_FitLAC0'
    lbl[21] = 'AHP_FitLDCRise0'
    lbl[22] = 'AHP_FitLDCFall0'
    lbl[23] = 'AHP_FitLAC1'
    lbl[24] = 'AHP_FitLDCRise1'
    lbl[25] = 'AHP_FitLDCFall1'
    lbl[26] = 'AHP_FitMeanAD' 
    lbl[27] = 'AHP_FitMaxAD' 
    self.lbl = lbl
  def analyse(self, _X = None, _si = 1., pgb = None):
    if _X is None: return
    if SHOWFITS: fig = iplot.figs()
    si = float(_si)
    n23 = int( (2*_X.shape[1]) / 3 ) # max and min must be within first 2/3 of vector
    MMA.analyse(self, _X[:,:n23])
    self.n = _X.shape[1]
    maxi, mini = np.array(self.Z[:,2], dtype = int), np.array(self.Z[:,3], dtype = int)
    self.Z = np.hstack((self.Z, np.empty((self.N, len(self.lbl)-self.Z.shape[1]), dtype = float)))
    self.D = diff0(self.Y[:,:n23], 2) # ascent and descent must be within first 2/3 of vector
    asci, desi = self.D.argmax(axis=1), self.D.argmin(axis=1)
    self.Z[:,4] = self.D[np.arange(self.N), asci] / si
    self.Z[:,5] = self.D[np.arange(self.N), desi] / si
    self.Z[:,6] = asci
    self.Z[:,8] = self.Y[np.arange(self.N), asci]
    self.Z[:,7] = desi
    self.Z[:,8] = self.Y[np.arange(self.N), asci]
    self.Z[:,9] = self.Y[np.arange(self.N), desi]
    if pgb is not None: pgb.init("Performing spike shape analysis", self.N)
    for i in range(self.N):
      if pgb is not None: pgb.set(i)
      y = self.Y[i]                       # raw waveform
      d = self.D[i]                       # waveform derative smoothed by order #2
      di = float(self.maxi[i] - asci[i])  # don't use descent - it's unreliable for noisy spikes
      J0 = int(asci[i] + self.ptht[0]*di) # self.ptht[0] should be < -1. but < -1.5 is safer for linear gradient
      J1 = int(asci[i] + self.ptht[1]*di) # self.ptht[1] should be > 0, but > 0.1 is safer for imposing curvature
      I0 = np.nonzero(y[:J0] > 2.*y[J0] - self.Z[i,4])[0] # just check we're still not a long way from threshold
      warn = True
      if len(I0):                         
        I0 = I0[-1]
        if I0 > asci[i] + self.ptht[0]*float(self.maxi[i] - J0):
          warn = False
          J0 = int(asci[i] + self.ptht[0]*float(self.maxi[i] - I0))
      if J0 < 0: # too close to threshold for reliable estimation
        if warn:
          print("spikeShape Warning: insufficient pre-threshold trace included for reliable threshold detection.")
        J0 = 0
      Jy = d[J0:J1]
      Jx = np.arange(J0, J1, dtype = float)
      Jp = htanfit(Jx, Jy)[0]
      ''' # Uncomment to see htan fits
      cla()
      plot(Jx, Jy, 'y')
      hold(True)
      plot(Jx, htanval(Jp, Jx), 'k')
      raw_input()
      '''
      j0 = int(round(self.pthc / expo(Jp[3]) + Jp[2]))
      j0 = max(J0, min(j0, J1)) # bound j0 to sensible values
      th = y[j0]

      self.Z[i,2] -= j0
      self.Z[i,3] -= j0
      self.Z[i,6] -= j0
      self.Z[i,7] -= j0
      self.Z[i,2:4] *= si
      self.Z[i,6:8] *= si
      self.Z[i,10] = th
      self.Z[i,11] = j0 * si
      hmax = 0.5 * (self.Z[i,0] + th)
      hmaxi = np.nonzero(y >= hmax)[0]
      hmax0, hmaxe = hmaxi[0], hmaxi[-1]
      self.Z[i,12] = (hmaxe - hmax0) * si
      je = np.nonzero(y >= th)[0][-1]
      if je > len(y)-8: # if the spike seems endless 
        je = int(j0 + (self.maxi[i] - j0)*2)
      self.Z[i,13] = (je - j0) * si
      jf = mini[i]
      ahp = self.Z[i,1] - th
      ahppol = +1
      More = np.greater
      if jf < j0: # if there is no AHP
        jf = je
        ahp = None
        More = np.less
        ahppol = -1
      yf = y[jf:]
      f, F = indfrac(yf, self.FCT, ahppol, True)
      Fm = np.median(yf[f[1]:])
      _fm = np.nonzero(More(yf, Fm))[0]
      fm = f[1]
      if len(_fm): fm = max(fm, _fm[0])
      f[2] = np.maximum(f[2], jf + 8)
      df = f[1] - f[0]
      dF = F[1] - F[0]
      fe = len(y) if ahp is None else max(je+8, f[2] + jf)
      '''
      print(th, ahppol, je, fe, len(y))
      cla()
      plot(y)
      raw_input()
      '''
      self.Z[i,14] = dF
      self.Z[i,15] = float(df) * si
      self.Z[i,16] = Fm
      self.Z[i,17] = float(fm + jf - j0) * si
      self.Z[i,18] = float(fe - je) * si
      ahpy = y[je:fe]
      ahpx = np.arange(len(ahpy), dtype = float) * si
      ahpp = np.tile(np.nan, 7)
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          _ahpp = exp2fit(ahpx, ahpy, 2)[0]
        ahpp = _ahpp
      except ValueError:
        print("spikeShape Warning: AHP fit error")
      ahph = exp2val(ahpp, ahpx)
      ahpd = np.fabs(ahpy - ahph)[1:] # the first point we ignore
      self.Z[i,19:26] = ahpp
      self.Z[i,26] = ahpd.mean()
      self.Z[i,27] = ahpd.max()
      if SHOWFITS:
        fig.newPlot()
        plot(ahpx, ahpy, 'c')
        hold(True)
        plot(ahpx, ahph, 'k')
        title(str(i))
    if pgb is not None: pgb.close()

class stepResponse(MMA):
  defPCT = [20, 80, 98]
  numlbl = 28
  ny0 = 256
  dx0 = -1
  def __init__(self, _X = None, _R = None, _si = 1.): # X = Passive, R = Active
    self.setCenSca()
    self.setLabel()
    self.setPCT(self.defPCT)
    self.analyse(_X, _R, _si)
  def setPCT(self, _PCT = None):
    if _PCT is not None: self.PCT = np.array(_PCT, dtype = float)    
    self.FCT = self.PCT * 0.01;
  def setLabel(self):
    lbl = [[]] * (self.numlbl)
    lbl[0] = 'Maximum'
    lbl[1] = 'Minimum' 
    lbl[2] = 'Maxtime'
    lbl[3] = 'Mintime' 
    lbl[4] = 'Basetime'
    lbl[5] = 'Steptime'
    lbl[6] = 'BaseMednCom'  
    lbl[7] = 'StepMednCom'
    lbl[8] = 'BaseMeanRec'
    lbl[9] = 'BaseStdvRec'
    lbl[10] = 'StepDis'
    lbl[11] = 'StepDisDelta'
    lbl[12] = 'IPR'
    lbl[13] = 'IPT'
    lbl[14] = 'Asymptote'
    lbl[15] = 'Asymptime'
    lbl[16] = 'Exp_FitDur'
    lbl[17] = 'Exp_FitOffset'
    lbl[18] = 'Exp_FitLACPos'
    lbl[19] = 'Exp_FitLDCPos'
    lbl[20] = 'Exp_FitLACNeg'
    lbl[21] = 'Exp_FitLDCNeg'
    lbl[22] = 'Exp_FitMeanAD'
    lbl[23] = 'Exp_FitMaxAD'
    lbl[24] = 'DeltaCom'
    lbl[25] = 'DeltaRec'
    lbl[26] = 'DRecDComQ'
    lbl[27] = 'Exp_FitC'
    self.lbl = lbl
  def analyse(self, _X = None, _C = None, _si = 1., pgb = None):
    if _X is None or _C is None: return
    if SHOWFITS: fig = iplot.figs()
    si = float(_si)
    MMA.analyse(self, _X)
    maxi, mini = np.array(self.Z[:,2], dtype = int), np.array(self.Z[:,3], dtype = int)
    self.Z = np.hstack((self.Z, np.empty((self.N, len(self.lbl)-self.Z.shape[1]), dtype = float)))
    self.prot = protocol()
    self.prot.setSamp(si)
    _ny0 = self.ny0
    done = False
    while not(done):
      y_, x0, y0, dx, Dy = self.prot.estProt(_C, _ny0)
      done = x0 > 0
      if not(done):
        done = _ny0 == 1
        _ny0 /= 2
    x0 += self.dx0
    dx = max(dx, 4)
    if self.dx0 > 0: dx -= self.dx0
    dx1 = int(dx/4)
    dx2 = int(dx/2)
    dx3 = dx1+dx2
    B = self.Y[:,:x0]
    M = np.mean(B, axis = 1)
    im = argmaxdis(self.Y[:, x0:], M)
    if pgb is not None: pgb.init("Performing step analysis", self.N)
    for i in range(self.N):
      if pgb is not None: pgb.set(i)
      _r = self.Y[i]
      _c = _C[i]
      r_, r = _r[:x0], _r[x0:(x0+dx-1)]
      c_, c = _c[:x0], _c[x0:(x0+dx-1)]
      cbase, cstep = np.median(c_), np.median(c)
      cdelta = cstep - cbase
      _rmn, _rsd = np.mean(r_), np.std(r_)
      t = np.arange(len(r), dtype = float) * si
      pol = 1. if np.mean(r[:dx1]) > np.mean(r[dx3:]) else -1.
      if pol > 0.:
        Less = np.less_equal
      else:
        Less = np.greater_equal
      h = max(3, min(len(r)-3, im))
      rh = r[h:]
      f, F = indfrac(rh, self.FCT, -pol, True)
      df = f[1] - f[0] 
      dF = F[0] - F[1]
      Fm = np.median(rh[f[1]:])
      _fm = np.nonzero(Less(rh, Fm))[0]
      fm = f[1]
      if len(_fm): fm = max(fm, _fm[0])
      fe = max(f[2], 5)
      rdelta = Fm - _rmn
      rq = rdelta / cdelta
      y = r[:(h+fe)]
      x = t[:(h+fe)]
      self.Z[i,2] -= x0
      self.Z[i,3] -= x0
      self.Z[i,4] = x0
      self.Z[i,5] = dx
      self.Z[i,2:6] *= si
      self.Z[i,6] = cbase
      self.Z[i,7] = cstep
      self.Z[i,8] = _rmn
      self.Z[i,9] = _rsd
      self.Z[i,10] = r[h]
      self.Z[i,11] = r[h] - _rmn
      self.Z[i,12] = dF
      self.Z[i,13] = float(df) * si
      self.Z[i,14] = Fm
      self.Z[i,15] = float(h+fm) * si
      self.Z[i,16] = float(fe) * si
      hp = np.tile(np.nan, 5)
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          _hp = exppfit(x, y, 1)[0]
        hp = _hp
      except ValueError:
        print("stepResponse Warning: exponential fit error")
      hy = exppval(hp, x)
      hd = np.minimum(np.fabs(hy - y), self.Z[i, 0]-self.Z[i, 1])
      sc = np.exp(-max(hp[-3], hp[-1])) / rq
      self.Z[i,17:22] = hp
      self.Z[i,22] = hd.mean()
      self.Z[i,23] = hd.max()
      self.Z[i,24] = cdelta
      self.Z[i,25] = rdelta
      self.Z[i,26] = rq
      self.Z[i,27] = sc
      if SHOWFITS:
        _t = si * (np.arange(len(r)) - h)
        fig.newPlot()
        plot(t, r, 'y')
        hold(True)
        plot(x, y, 'c')
        plot(x, hy, 'k')
        title(str(i))
    if pgb is not None: pgb.close()

class stepBiexp(MMA):
  defPCT = [20, 80, 98]
  numlbl = 28
  ny0 = 256
  dx0 = -1
  def __init__(self, _X = None, _R = None, _si = 1.): # X = Passive, R = Active
    self.setCenSca()
    self.setLabel()
    self.setPCT(self.defPCT)
    self.analyse(_X, _R, _si)
  def setPCT(self, _PCT = None):
    if _PCT is not None: self.PCT = np.array(_PCT, dtype = float)    
    self.FCT = self.PCT * 0.01;
  def setLabel(self):
    lbl = [[]] * (self.numlbl)
    lbl[0] = 'Maximum'
    lbl[1] = 'Minimum' 
    lbl[2] = 'Maxtime'
    lbl[3] = 'Mintime' 
    lbl[4] = 'Basetime'
    lbl[5] = 'Steptime'
    lbl[6] = 'BaseMednCom'  
    lbl[7] = 'StepMednCom'
    lbl[8] = 'BaseMeanRec'
    lbl[9] = 'BaseStdvRec'
    lbl[10] = 'StepDis'
    lbl[11] = 'StepDisDelta'
    lbl[12] = 'IPR'
    lbl[13] = 'IPT'
    lbl[14] = 'Asymptote'
    lbl[15] = 'Asymptime'
    lbl[16] = 'Exp_FitDur'
    lbl[17] = 'Exp_FitOffset'
    lbl[18] = 'Exp_FitAmp0'
    lbl[19] = 'Exp_FitLDC0'
    lbl[20] = 'Exp_FitAmp1'
    lbl[21] = 'Exp_FitLDC1'
    lbl[22] = 'Exp_FitMeanAD'
    lbl[23] = 'Exp_FitMaxAD'
    lbl[24] = 'DeltaCom'
    lbl[25] = 'DeltaRec'
    lbl[26] = 'DComDRecQ'
    lbl[27] = 'Exp_FitC'
    self.lbl = lbl
  def analyse(self, _X = None, _C = None, _si = 1., pgb = None):
    if _X is None or _C is None: return
    if SHOWFITS: fig = iplot.figs()
    si = float(_si)
    MMA.analyse(self, _X)
    maxi, mini = np.array(self.Z[:,2], dtype = int), np.array(self.Z[:,3], dtype = int)
    self.Z = np.hstack((self.Z, np.empty((self.N, len(self.lbl)-self.Z.shape[1]), dtype = float)))
    self.prot = protocol()
    self.prot.setSamp(si)
    _ny0 = self.ny0
    done = False
    while not(done):
      y_, x0, y0, dx, Dy = self.prot.estProt(_C, _ny0)
      done = x0 > 0
      if not(done):
        done = _ny0 == 1
        _ny0 /= 2
    x0 += self.dx0
    dx = max(dx, 4)
    if self.dx0 > 0: dx -= self.dx0
    dx1 = int(dx/4)
    dx2 = int(dx/2)
    dx3 = dx1+dx2
    B = self.Y[:,:x0]
    M = np.mean(B, axis = 1)
    im = argmaxdis(self.Y[:, x0:], M)
    if pgb is not None: pgb.init("Performing step analysis", self.N)
    for i in range(self.N):
      if pgb is not None: pgb.set(i)
      _r = self.Y[i]
      _c = _C[i]
      r_, r = _r[:x0], _r[x0:(x0+dx-1)]
      c_, c = _c[:x0], _c[x0:(x0+dx-1)]
      cbase, cstep = np.median(c_), np.median(c)
      cdelta = cstep - cbase
      _rmn, _rsd = np.mean(r_), np.std(r_)
      t = np.arange(len(r), dtype = float) * si
      pol = 1. if np.mean(r[:dx1]) > np.mean(r[dx3:]) else -1.
      if pol > 0.:
        Less = np.less_equal
      else:
        Less = np.greater_equal
      h = im
      rh = r[h:]
      f, F = indfrac(rh, self.FCT, -pol, True)
      df = f[1] - f[0] 
      dF = F[0] - F[1]
      Fm = np.median(rh[f[1]:])
      _fm = np.nonzero(Less(rh, Fm))[0]
      fm = f[1]
      if len(_fm): fm = max(fm, _fm[0])
      fe = max(f[2], 5)
      rdelta = Fm - _rmn
      rq = cdelta / rdelta
      y = r[h:(h+fe)]
      x = t[h:(h+fe)]
      self.Z[i,2] -= x0
      self.Z[i,3] -= x0
      self.Z[i,4] = x0
      self.Z[i,5] = dx
      self.Z[i,2:6] *= si
      self.Z[i,6] = cbase
      self.Z[i,7] = cstep
      self.Z[i,8] = _rmn
      self.Z[i,9] = _rsd
      self.Z[i,10] = r[h]
      self.Z[i,11] = r[h] - _rmn
      self.Z[i,12] = dF
      self.Z[i,13] = float(df) * si
      self.Z[i,14] = Fm
      self.Z[i,15] = float(h+fm) * si
      self.Z[i,16] = float(fe) * si
      hp = np.tile(np.nan, 5)
      try:
        with  warnings.catch_warnings():
          warnings.simplefilter("ignore")
          _hp = expdfit(x, y, 2)[0]
        hp = _hp
      except ValueError:
        print("stepBiexp Warning: exponential fit error")
      hy = expdval(hp, x)
      hd = np.minimum(np.fabs(hy - y), self.Z[i, 0]-self.Z[i, 1])
      sc = np.exp(-min(hp[-3], hp[-1])) / rq
      self.Z[i,17:22] = hp
      self.Z[i,22] = hd.mean()
      self.Z[i,23] = hd.max()
      self.Z[i,24] = cdelta
      self.Z[i,25] = rdelta
      self.Z[i,26] = rq
      self.Z[i,27] = sc
      if SHOWFITS:
        _t = si * (np.arange(len(r)) - h)
        fig.newPlot()
        plot(x, y, 'y')
        hold(True)
        plot(x, hy, 'k')
        title(str(i))
    if pgb is not None: pgb.close()

class expDecay (EEE): # only modification: __init__ does not include centrescale
  def __init__(self, _X = None, _si = 1.):
    self.setCenSca()
    self.setLabel()
    self.setPCT(self.defPCT)
    self.analyse(_X, _si)

class synResponse(RFA):
  def __init__(self, _X = None, si = None, irange = None, _u = None):    
    self.mwd = MWD()
    self.setCenSca()
    self.setPCT(self.defPCT)
    self.setLabel()
    self.analyse(_X, si, irange, _u)    
  def setLabel(self):
    RFA.setLabel(self)
    self.lbl += self.mwd.lbl
  def analyse(self, _X = None, si = 1., irange = None, _u = None, pgb = None):
    if _X is None: return
    RFA.analyse(self, _X, si, _u, pgb) # Note that self.Z is already final size
    self.mwd.analyse(_X, irange, _u)
    self.Z[:,-self.mwd.Z.shape[1]:] = self.mwd.Z
    self.Z = np.hstack( (self.Z[:,:-self.mwd.Z.shape[1]], self.mwd.Z) )

