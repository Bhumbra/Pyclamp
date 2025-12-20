#!/usr/bin/python3
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pyclamp'))
sys.path.append(os.path.join(os.getcwd(), '../probayes'))
from pyclamp import pyclamp

if __name__ == '__main__':
  self = pyclamp.main()
  sys.exit(self.App.Widget.exec())

