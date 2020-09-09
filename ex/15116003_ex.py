import sys
sys.path.append("/home/admin/code/python/pyclamp")
sys.path.append("/home/admin/code/python/quantal")

from mplqt import *
from pyclamp import pyclamp

I = [100, 118, 160]
dr = "/home/admin/data/"
si = 5e-05

self = pyclamp()
self.setDataDir(dr)

self.setDataFiles(['15116003.abf'])
self.setChannels([0, 1], [[5e-05, 0.61035153351, 0.0], [5e-05, 0.61035153351, 0.0]])
self.readIntData()
self.setSelection([-291])
self.setClampConfig(1, 0)
self.trim(0, [68026, 68445], True)
self.setActive([[161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
  182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204],
  True], 291)
self.align(0, -1, [103, 130], False)
self.baseline(0, [0, 50])
x = self.extract(0)

X = [x[170:210,I[0]:I[1]], x[170:210, I[1]:I[2]]]
minx = X[0].min(axis=1)
maxxv = X[1].max(axis=1)
maxxi = X[1].argmax(axis=1)
minxv = X[1].min(axis=1)
minxi = X[1].argmin(axis=1)

figure()
subplot(2, 2, 1)
plot(minx, maxxv, '.')
xlabel('Trough')
ylabel('Maximum')

subplot(2, 2, 2)
plot(minx, maxxi*si, '.')
xlabel('Trough')
ylabel('Maxtime')

subplot(2, 2, 3)
plot(minx, minxv, '.')
xlabel('Trough')
ylabel('Minimum')

subplot(2, 2, 4)
plot(minx, minxi*si, '.')
xlabel('Trough')
ylabel('Mintime')


