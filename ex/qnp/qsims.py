# execfile("qsims.py")

# A multiple quantal simulator

import numpy as np
from numpy import *
from pyclamp.gui.mplqt import *
from pyclamp.dsp.iofunc import *
from pyclamp.dsp.fpfunc import *
from pyclamp.guiiplot import pbfig

import qsim

# PARAMETERS

output = True
dirn = '/home/admin/data/Q/tdf/'
extn = '.tdf'
q = 100.
useGamma = [False, True]
nsims = 100 # 1000

typnNmsSueva = [int, int, int, float, float, float, int, float, float]
geonNmsSueva = [0, 0, 0, 0, 0, 1, 1, 1, 1]
strnNmsSueva = ['n', 'N', 'm', 's', 'S', 'u', 'e', 'v', 'a']
lblnNmsSueva = [r'$n$', r'$N$', r'$\vert \mu \vert$', r'$s_0$', r'$\Delta s$', r'$u$', r'$e$', r'$v$', r'$a$', 'Repetitions']

defnNmsSueva = [6, 60,  3, 0.1, 0.5,  0.3,  25,    0,  NaN]
minnNmsSueva = [1, 30,  3, 0.1, 0.05, 0.01,  1, 0.01,  0.2]
maxnNmsSueva = [16, 120, 9, 0.4, 0.8,  0.64, 64, 0.64,  12.8]
resnNmsSueva = [16, 46, 7, 31, 76, 7, 7, 7, 7]
varnNmsSueva = [False, False, False, False, False, False, False, False, False]
forceNoSubDir = False
forcenNmSuStr = False
notOverWriteOutput = True

defnNmsSueva = [30, 60,  2, 0.1, -2.,  0.3,  20,    0,  NaN]
minnNmsSueva = [1, 30,  3, 0.05, 0.05, 0.01,  1, 0.01,  0.2]
maxnNmsSueva = [16, 120, 9, 0.45, 0.8,  0.64, 64, 0.64,  12.8]
resnNmsSueva = [16, 46, 7, 9, 16, 7, 7, 7, 7]
varnNmsSueva = [False, False, False, True, False, False, False, False, False]
forceNoSubDir = False
forcenNmSuStr = False
notOverWriteOutput = True

# INPUT 

strnnmssueva = array(strnNmsSueva)[array(varnNmsSueva)].tostring()

nv = len(defnNmsSueva)
nNmsSueva = [[]] * nv
M = [[]] * (nv+1)

for i in range(nv):
  if varnNmsSueva[i]:
    nNmsSueva[i] = numspace(minnNmsSueva[i], maxnNmsSueva[i], resnNmsSueva[i], geonNmsSueva[i])    
    if typnNmsSueva[i] is int:
      nNmsSueva[i] = array(np.round(nNmsSueva[i]), dtype = typnNmsSueva[i])
    else:  
      nNmsSueva[i] = array(nNmsSueva[i], dtype = typnNmsSueva[i])
    M[i] = resnNmsSueva[i]
  else:  
    nNmsSueva[i] = defnNmsSueva[i]
    nNmsSueva[i] = array(nNmsSueva[i], dtype = typnNmsSueva[i]).reshape(1)  
    M[i] = 1
M[-1] = nsims

# PROCESSING AND OUTPUT

opdir = dirn

if not(forceNoSubDir):
  opdir += strnnmssueva
  if output:
    mkdir(opdir)  
  opdir +=  '/' 

pb = pbfig("Undertaking simulations")
pb.setup(lblnNmsSueva, M)

x = [[]] * nv
Q = [[]] * nsims

for i0 in range(M[0]):
  x[0] = nNmsSueva[0][i0]
  for i1 in range(M[1]):
    x[1] = nNmsSueva[1][i1]
    for i2 in range(M[2]):
      x[2] = nNmsSueva[2][i2]  
      for i3 in range(M[3]):
        x[3] = nNmsSueva[3][i3]        
        for i4 in range(M[4]):
          x[4] = nNmsSueva[4][i4]
          X = x[3] + x[4] if x[4] >= 0. else -x[3]*x[4]
          s = linspace(x[3], X, x[2])
          for i5 in range(M[5]):
            x[5] = nNmsSueva[5][i5]        
            for i6 in range(M[6]):
              x[6] = nNmsSueva[6][i6]
              for i7 in range(M[7]):
                x[7] = nNmsSueva[7][i7]
                for i8 in range(M[8]):
                  x[8] = nNmsSueva[8][i8]                  
                  pb.update([i0, i1, i2, i3, i4, i5, i6, i7, i8, 0])
                  filn = ''
                  if forcenNmSuStr:
                    filn += strnNmsSueva[1] + str(x[1])
                    filn += strnNmsSueva[2] + str(x[2])
                    filn += strnNmsSueva[4] + str(x[4])
                    filn += strnNmsSueva[5] + str(x[5])
                  else:
                    for i in range(nv):
                      if varnNmsSueva[i]:
                        filn += strnNmsSueva[i] + str(x[i])     
                  if notOverWriteOutput and os.path.isfile(opdir + filn + extn):
                    pass
                  else:
                    if isnan(x[8]):
                      qm = qsim.binoqsim(x[6], x[0], q, x[5], x[7], s[0], useGamma)                
                    else:
                      qm = qsim.betaqsim(x[6], x[0], q, x[5], x[7], x[8], s[0], useGamma)
                    for i in range(nsims):
                      pb.update([i0, i1, i2, i3, i4, i5, i6, i7, i8, i])
                      Q[i] = qm.simulate(x[1], s)
                    if output:
                      writeDTFile(opdir + filn + extn, Q)                        
pb.close()          
