# A pyclamp batch analyser

import sys
import os
from multiprocessing import Process
cwdir = os.getcwd()
sys.path.append(cwdir + '/../pyclamp/')
from iofunc import *

def f(x, abfdir, writeop):
  exec("from pyclamp import *")
  exec("self = pyclamp()")
  exec("".join(("self.setDataDir('", abfdir, "')")))
  for k in range(len(x)-1):
    exec("".join(("self.", x[k])))
  if writeop:
    k = len(x)-1
    exec("".join(("self.", x[k])))

ipdir = '/home/admin/analyses/old/'
ipext = '.tab'
abfdir = '/home/admin/data/Single_GlyT2/'
writeop = True

# INPUT

_ipfils = os.listdir(ipdir)

ipfils = []

for i in range(len(_ipfils)):
  stem, extn = os.path.splitext(_ipfils[i])
  if extn.lower() == ipext:
    ipfils.append(_ipfils[i])
                  
n = len(ipfils)

for i in range(n):
  print(str(i) + ': ' + ipfils[i])
  
guid = raw_input("Select file index to start (or just press enter for all batch analysis): ") 

if len(guid):
  ipfils = ipfils[int(guid):]

n = len(ipfils)


for i in range(n):
  X = readDTFile(ipdir+ipfils[i])
  nX = len(X)
  if nX % 2: print("Warning: odd data contents in file: " + ipfils[i])
  for j in range(0, nX, 2):
    print("Processing file #" + str(i+1) + "/" + str(n) + " (" + ipfils[i] + "), log #" + str(j/2+1) + "/" + str(nX/2))
    p = Process(target=f, args=(X[j], abfdir, writeop))
    p.start()
    p.join()
    p.terminate()
    del p

