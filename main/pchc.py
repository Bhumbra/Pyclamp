import sys
import pyqtgraph as pg
import pyplot
pyplot.SetJetRGBC(1.5)
pyplot.SetDefScatMarkerSize(6)
import pyclamp

if __name__ == '__main__':
  self = pyclamp.main()
  sys.exit(self.App.Widget.exec_())

