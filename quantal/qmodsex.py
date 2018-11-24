import sys
import qmods

qmods.DEFIPD = '/home/admin/analyses/mnrc/vrrc/tsv2/'

EHAT = -1.
SRES = 128
VRES = 64
ARES = 1
NRES = 32
NMAX = 249

def main(_argv):
  argv = [EHAT, SRES, VRES, ARES, NRES, NMAX]
  k = 0
  for i in range(len(_argv)): 
    if _argv[i].find('.py') < 0:
      argv[k] = int(_argv[i]) if k else float(_argv[i])
      k += 1
  qmods.qmods(float(argv[0]), argv[1], argv[2], argv[3], argv[4], argv[5])

if __name__ == "__main__": 
  main(sys.argv)

