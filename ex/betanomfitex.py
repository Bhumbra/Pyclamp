# execfile("betanomfit.py")

# beta-nomial fit

from pylab import *
from prob import *
import scipy.stats as stats
import iplot

mino = 1e-300
onem = 1e-11

# PARAMETERS

Nn = 4
N = 64

nlims = [2, 16]
milims = [0.2, 0.8]
ailims = [0.01, 0.01]

estimate = False

# INPUT

n = ceil(geospace(nlims[0], nlims[1], Nn))
ai = linspace(ailims[0], ailims[1], N)
mi = linspace(milims[0], milims[1], N)
bi = ai/mi - ai
vi = (ai*bi) / (ai+bi)**2.0 / (ai+bi+1.)
si = sqrt(vi)
i3 = 2.*(bi-ai)**2.0*sqrt(ai+bi+1.)/(ai+bi+2.)/sqrt(ai*bi)
#ei = log(beta(ai, bi)) - (ai - 1.) * special.digamma(ai) - (bi - 1.)*special.digamma(bi) + (ai+bi-2.)*special.digamma(ai+bi)
#ei = log(beta(ai, bi)) - (ai - 1.) * special.digamma(ai) - (bi - 1.)*special.digamma(bi) + (ai+bi-2.)*special.digamma(ai+bi)
ei = betaentropy(ai, bi)

# PROCESSING

P = [[]] * Nn
Eb = [[]] * Nn

pb = iplot.pbfig(r"$n$")
pb.setup(['$n$', r'$\mu$'], [Nn, N], ['r', 'b'])

for i in range(Nn):
  pb.forceupdate([i, 0])
  P[i] = betanompmf(n[i], ai, bi, pb)
  I = tile(arange(n[i]+1).reshape(1, n[i]+1), (N, 1))
  Mi = tile(mi.reshape(N, 1), (1, n[i]+1))
  BP = stats.binom.pmf(I, n[i], Mi)
  Eb[i] = sum(-BP * log(BP+mino), axis = 1)
pb.close()


X = [[]] * Nn

Ao = empty((Nn, N), dtype = float)
Bo = empty((Nn, N), dtype = float)
Mo = empty((Nn, N), dtype = float)
No = empty((Nn, N), dtype = float)
Vo = empty((Nn, N), dtype = float)
Eo = empty((Nn, N), dtype = float)
Vb = empty((Nn, N), dtype = float)
Pf = empty((Nn, 2), dtype = float)

for i in range(Nn):
  X[i] = linspace(onem, 1. - onem, n[i]+1)
  if not(estimate):
    Mo[i, :] = sum(P[i]*X[i], axis = 1)
    #Vb[i, :] = float(n[i])*Mo[i, :]*(1.-Mo[i, :])
    Vo[i, :] = sum(P[i]*(tile(X[i].reshape((1, n[i]+1)), (N, 1))-tile(Mo[i, :].reshape( (N, 1) ), [1, n[i]+1]))**2.0 , axis = 1)
    No[i, :] = Mo[i, :] * (1. - Mo[i, :]) / Vo[i, :] - 1.0
    Ao[i, :] = No[i, :] * Mo[i, :]
    Bo[i, :] = Ao[i, :]/Mo[i, :] - Ao[i, :]
    Eo[i, :] = betaentropy(Ao[i, :], Bo[i, :])
    Pf[i, :] = polyfit(log(Mo[i, :]), log(Vo[i, :]/vi), 1) 
  else:  
    Mo[i, :] = np.tile(mi[i].reshape( (1, N) ), [n[i]+1, 1])
    '''
    Ao[i][j] = 1. / (bi[i] * (float(n) - 1.))
    Bo[i][j] = ao[i]/mo[i] - ao[i]
    Vo[i][j] = (ao[i]*bo[i]) / (ao[i]+bo[i])**2.0 / (ao[i]+bo[i]+1.)
    Pf[i, :] = polyfit(log(Mo[i, :]), log(Vo[i, :]/Vi[i, :]), 1) 
    '''
  
nr = tile(n.reshape( (Nn, 1) ), [1, 2])
    
# OUTPUT  

nr = floor(sqrt(N))
nc = ceil(N/nr)
close('all')

for i in range(N):
  subplot(nr, nc, i+1)
  plot(X[0], P[0][i])
  hp = 1./float(n[0]+1) * stats.beta.pdf(X[0], Ao[0,i], Bo[0,i])
  plot(X[0], hp/hp.sum())
       
figure()  
plot(log(mi),log(Vo[0,:]/vi))

figure()
plot(n, Pf[:,0])

de = ei-Eo[0]-Eb[0]

