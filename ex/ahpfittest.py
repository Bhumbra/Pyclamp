ifrom matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


from pyclamp import pyclamp
from wcicanal import *
from optfunc import *
import iwave as iw
import pylab as pl

dr = "/home/admin/Code/Python/pyclamp"


mark = pl.array([2, 50000])
pc = pyclamp()
pc.setDataDir(dr)

pc.setDataFiles(['100623n5_0005.abf'])
pc.setChannels([1, 0], [[2.0000000000000002e-05, 0.030517576675500001, 0.0], [2.0000000000000002e-05, 0.00030517578125, 0.0]])
pc.readIntData()
pc.setSelection([[0]])
pc.setClampConfig(0, 0)
pc.trim(0, [5587, 100955])
pc.trigger(0, 1, -19.3341771499)
pc.align(0)
pc.trim(0, [11247, 18159])
pc.trim(0, [2891, 5506])

#wav = iw.imulwav(pc.Data, 2e-5)
#wav.setPlots()
#pl.show()

ilo = 580

X = pc.Data[0]
dx = 0.00

i0 = pl.arange(4)
i1 = pl.array( (0, 4, 5, 6) )

for i in range(8):
  y = X[i][ilo:]  
  x = pl.linspace(dx, dx +(i+len(y))* 2e-5, len(y))
  h = exppfit(x, y, 2)
  hp = h[0]
  hp0 = hp[i0]  
  hp1 = hp[i1]
  hp1[0] = y[0]
  infx = exppinf(hp)
  hy = exppval(hp, x)
  hy0 = exppval(hp0, x)
  hy1 = exppval(hp1, x)
  
  #ss = spikeShape(2.0000000000000002e-05)
  #ss.analyse(X)
  ax = pl.subplot(2, 4, i+1)
  pl.plot(x, y, 'c')
  pl.hold(True)
  pl.plot(x, hy, 'k')
  pl.plot(x, hy0, 'g')
  pl.plot(x, hy1, 'r')
  tau = [1.0 / pl.exp(hp1[3]), 1.0 / pl.exp(hp0[3])]
  ax.xaxis.set_major_locator(pl.MaxNLocator(4))
  miny = y.min()
  pl.ylim([miny, miny+7.0])              
  #pl.title(''.join([r'$\Lambda_0=', str(hp1[3]), '$, ', r'$\Lambda_1=', str(hp0[3]), '$']))
  pl.title(''.join([r'$\alpha_0=', str(hp1[1]), '$, ', r'$\alpha_1=', str(hp0[1]), '$']))

pl.show()
print(hp)
