from pgb import *
from time import time, sleep


n1 = 10
n2 = 20
dt = 0.05
if DISPLAY: qapp = QAPP([])


#'''
pb = pgb()
pb.init("Title", ['1', '2'], [n1, n2], ['r', 'b'])

t = time()
for i in range(n1):
  j = 0
  pb.set([i, j])
  for j in range(n2):
    pb.set(j)
    sleep(dt)   
  
pb.close()
#'''

'''
 
pb = pgb()
pb.init('1', n1*n2, 'b')

t = time()
for i in range(n1*n2):
  pb.set(i)
  sleep(dt)   
  
pb.close()
'''
print("Time taken (s): " + str(time() - t))

