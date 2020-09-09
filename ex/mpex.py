import numpy as np
import multiprocessing as mp
import time

def f(q):
  q.put(np.empty( (1000, 1000, 200), dtype = float))

Q = mp.Queue()
P = mp.Process(target = f, args=(Q,))
P.start()
X = Q.get()
#P.join()
#del X

print("created")
for i in range(200000):
  time.sleep(0.000001)
del X
print("deleted")
for i in range(200000):
  time.sleep(0.000001)

