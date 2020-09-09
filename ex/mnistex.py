import mnist_loader
from nnsup import *

# PARAMETERS

ne = 10
bs = 10
useReLU = False
useXEnt = True 
eta = 0.3
#lmbd = 0.
lmbd = 5.
#mu = 0.2
mu = 0.
#drop = [0.5, 0.]
drop = None
L = 1
#Arch = [784, 30, 30, 10]
Arch = [784, 30, 10]
#Arch = [784, 10]
testData = True 
testDataMax = np.inf
#testDataMax = 100
trainingDataMax = np.inf
#trainingDataMax = 1000

# INPUT

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
imax = min(trainingDataMax, len(training_data))
xyt = list2Dtr(training_data[:imax])
xt, yt = np.array(xyt[0]).T, np.array(xyt[1]).T
imax = min(testDataMax, len(test_data))
xyT = list2Dtr(test_data[:imax])
xT, yT = np.array(xyT[0]).T, np.array(xyT[1]).T
if testData is not None:
  if testData:
    x, K = xT, yT
    K = np.ravel(K)
    y = np.zeros( (yt.shape[0], yT.shape[1]), dtype = float)
    for i in range(len(K)):
      y[K[i], i] = 1.
  else:
    x, y = xt, yt
    K = np.argmax(y, axis = 0)

# PROCESSING

self = supebp(Arch)
self.setL(lmbd, L)
if useReLU: self.setA(ReLU, ReDU)
if useXEnt: self.setC(xentcf, xentcd)
CF = np.empty(ne, dtype = float)
NM = np.empty(ne, dtype = int)

for i in range(ne):
  self.supervise(xt, yt, 1, bs, eta, lmbd, mu, drop)
  if testData is not None:
    a = self.forward(x)
    CF[i] = self.backward(y)
    k = np.argmax(a, axis = 0)
    num = (k == K).sum()
    dem = len(K)
    NM[i] = num
    print("Epoch " + str(i) + ": " + str(num) + "/" + str(dem))


Du = self.Du if type(self.Du) is not list else self.Du[0]
DW = self.DW if type(self.DW) is not list else self.DW[0]

# OUTPUT

subplot(2, 2, 1)
plot(CF)

subplot(2, 2, 2)
plot(NM)

subplot(2, 2, 3)
plot(DW)
#yscale('log')

subplot(2, 2, 4)
plot(Du)
#yscale('log')


