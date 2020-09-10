import sys
from pyclamp import pyclamp

if __name__ == '__main__':
  self = pyclamp.main()
  sys.exit(self.App.Widget.exec_())

