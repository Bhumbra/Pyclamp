
from iplot import *
from pylab import *; ion()
from time import time, sleep


n1 = 10
n2 = 20
dt = 0.01

#figure()
#show(block=False)

pb = pbfig("Title")
#pb.useConsole()
pb.setup(['1', '2'], [n1, n2], ['r', 'b'])

t = time()
for i in range(n1):
  j = 0
  pb.update([i, j])
  for j in range(n2):
    pb.updatelast(j)
    sleep(dt)   
  
pb.close()

print "Time taken (s): " + str(time() - t)

