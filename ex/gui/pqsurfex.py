import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore
from pyqtplot import *
from pyplot import *
from fpfunc import *
from prob import *

width, height = 800, 600
n = 50
xy = np.arange(n)
fn = float(n)
zx = numspace(-3, 3, n)
z = np.matrix(zpdf(zx))
Z = np.array(z.T * z)
Z /= Z.max()

app = QtGui.QApplication([])
Form = BaseFormClass()
Area = area()
Form.setCentralWidget(Area)
Dock = dock()
Area.add(Dock)
Vbox = Dock.addVbox()
Surf = surf(zx, zx, Z, shader='normalColor')
Vbox.add(Surf)
Form.resize(width, height)
Form.show()

#pos = np.array([0.0, 0.5, 1.0])
#color = np.array([[0,0,0,255], [255,128,0,255], [255,255,0,255]], dtype=np.ubyte)
#cmap = pg.ColorMap(pos, color)
#lut = cmap.getLookupTable(0.0, 1.0, 256)
#ms.shader()['colorMap'] = lut
#view = BaseVboxClass()
#view.addItem(ms)
#view.show()

nframes = 360
Anim = anim(Vbox)
Anim.setAnimFunc(Vbox.orbit, 360./nframes, 90./nframes)
Anim.animate("test", nframes)
Anim.saveAnim(width, height)

