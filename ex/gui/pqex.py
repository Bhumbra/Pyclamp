from pyqtgraph import *
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.MeshData import *
from pyqtplot import *
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setCameraPosition(distance=10, azimuth=-90., elevation = 30.)
mid = np.array([0., 0., -3.])
'''
md = cylinder(1, 100, [0.5, 1.5], length=2)
mi = gl.GLMeshItem(meshdata=md, color=(1, 0, 0, 1))
mi.scale(1.3, 0.7, 1.)
mi.translate(0.5, 0., 1.)
mi.rotate(10., 10., 10., 10.)
'''
#tc0 = truncone((-1, -1, -0.2), (0, 0, 0), shader = 'shaded', color = (1, 0., 0., 1))
#tc1 = truncone((0, 0, 0), (1, 1, -0.2), shader = 'shaded', color = (0, 1., 0., 1))
tc0 = truncone(mid+np.array((0, 0, -0.3)), mid+np.array((0, 0, -0.1)), shader = 'shaded', color = (1, 0., 0., 1))
tc1 = truncone(mid+np.array((0, 0, 0.)), mid+np.array((0, 0, 0.2)), shader = 'shaded', color = (0, 1., 0., 1))
w.addItem(tc0)
w.addItem(tc1)

