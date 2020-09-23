from mplqt import *
from fpfunc import *
import numpy as np

lran = [-40, 20]
N = 20000
n = 1000
E = 10
dtlm = 0
dtls = 1.5
l = 0.0005

ldt = np.random.normal(dtlm, dtls, N)
dt = np.exp(ldt)
ldc = np.linspace(lran[0], lran[1], n)
s = np.empty(n, dtype = float)

for i in range(n):
  a = np.exp(-dt * np.exp(ldc[i]))
  s[i] = np.std(a)

i = np.argmax(s)
ldcsd = ldc[i]
asd = np.exp(-dt * np.exp(ldcsd))
sd = np.std(a)

m = -np.mean(ldt)
M = np.empty(N+1, dtype = float)
M[0] = m

for i in range(N):
  dm = 1. - dt[i] * np.exp(m)
  m += l * dm
  M[i+1] = m

ldcim = m
aim = np.exp(-dt * np.exp(ldcim))

nb = int(np.round(np.sqrt(N)))
b = linspace(0., 1., nb)
fsd = freqhist(asd, b)
fim = freqhist(aim, b)
psd = fsd / float(N)
pim = fim / float(N)
esd = -np.sum(psd * np.log(psd + 1e-300))
eim = -np.sum(pim * np.log(pim + 1e-300))

figure()
subplot(2, 2, 1)
plot(ldc, s)

subplot(2, 2, 2)
hist(asd, nb)

subplot(2, 2, 3)
plot(M)

subplot(2, 2, 4)
hist(aim, nb)

