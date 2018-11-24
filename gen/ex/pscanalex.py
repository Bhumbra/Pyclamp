#execfile("pscanal.py")

# IMPORTS

import matplotlib as mpl
#mpl.use('WXAgg')
mpl.use('Qt4Agg')
#mpl.rcParams['backend.qt4']='PySide'

from iplot import *
from fpanal import *
from wffunc import *
from iofunc import *
from pylab import *; ion()

import abf as md
import iwave as iw

# PARAMETERS

ipdn = '/home/admin/data/pairs/ABF/'
ipfn = '101029p2_00Spike1Quantal.abf'
opdn = '/home/admin/data/pairs/ABF/'
opdn = ipdn
usechan = 0; # channel index
tstart = 0.0045 # secs
tfinish = 0.015 #inf # secs
astart = 0.0000 # secs - use astart = None to not baseline
afinish = 0.0005 # secs
qsrange = 0.0002 # time window to calcuate quantal size around `peak' [use None for default]
qamplbl = 'QAmp'

#filt = [False, 200., 10000., 10] # [filt[0] (0 is on, set None for off), corner freq, stop freq., non-lin]
filt = [True, 100., 300., 2] # [filt[0] (0 is on, set None for off), corner freq, stop freq., non-lin]


# [mode None = off, mode 0 = LPF, mode 1 = HPF]

# INPUT

mdf = md.ABF(ipdn+ipfn)
data = mdf.ReadIntData()
chan = mdf.ReadChannelInfo()
si = chan[usechan].samplint
gn = chan[usechan].gain
of = chan[usechan].offset

# PROCESSING

D = data[usechan]
ns = D.shape[1]
istart = max(0, round(tstart / si))
ifinish = min(ns, round(tfinish / si))
D = D[:,istart:ifinish] * gn + of
chan[usechan].gain, chan[usechan].offset = 1, 0

ns = D.shape[1]
D += 100.
if filt[0]:
  X, ff = hpfilter(D, 1./si, [filt[1], filt[2]], filt[3], None, True)
else:
  X, ff = lpfilter(D, 1./si, [filt[1], filt[2]], filt[3], None, True)

figure()
subplot(2, 1, 1)
plot(D.T)
subplot(2, 1, 2)
plot(X.T)

figure()
subplot(2, 1, 1)
plot(ff.f, ff.fm)
subplot(2, 1, 2)
plot(ff.F, ff.Filter, '-')


