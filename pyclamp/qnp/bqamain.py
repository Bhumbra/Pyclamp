import sys
import qmods

def main(_argv):
  argv = [1, 128, 64, 1, 32, 32, 1, 0]
  k = 0
  for i in range(len(_argv)): 
    if _argv[i].find('.py') < 0:
      argv[k] = int(_argv[i]) if k else float(_argv[i])
      k += 1
  qmods.qmods(float(argv[0]), argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7])

if __name__ == "__main__": 
  main(sys.argv)

