from pyqtplot import *
from pyplot import jetrgb
from iofunc import *
import numpy as np

# PARAMETERS

ipdn = "/home/admin/loc/shows/npps/tsv/"
slfn = "101029p2_msl.tsv"
smfn = "101029p2_sl.tsv"

# INPUT

RSMDD = readSVFile(ipdn+slfn)
SM = readSVFile(ipdn+smfn)

# PROCESSING

RS = RSMDD[0]
MS = np.array([np.array(RSMDD[1]), np.array(RSMDD[2]), np.array(RSMDD[3])])
DD = [RSMDD[4], RSMDD[5], RSMDD[6]] 
#ddmax = np.max(np.ravel(np.array(DD)))

smx = [SM[0], SM[1], SM[2]]
smy = [SM[3], SM[4], SM[5]]

# OUTPUT

app = QtGui.QApplication([])
Form = BaseFormClass()
Area = area()
Form.setCentralWidget(Area)
Dock = dock()
Area.add(Dock)
Vbox = Dock.addVbox()
Vbox.setCameraPosition(distance = 1.8, elevation = 90., azimuth = -90.)
Form.show()

minx, maxx = np.inf, -np.inf 
miny, maxy = np.inf, -np.inf
maxdd = -np.inf
for h in range(len(MS)):
  for i in range(len(MS[h])):
    k = 0
    ms = np.ravel(MS[h][i])
    n = len(ms)/15
    dd = np.ravel(DD[h][i])
    maxdd = max(maxdd, max(dd))
    for j in range(n):
      mshij = ms[k:k+15]
      minx = min(minx, min(mshij[0:2]))
      maxx = max(maxx, max(mshij[0:2]))
      miny = min(miny, min(mshij[2:4]))
      maxy = max(maxy, max(mshij[2:4]))
      k += 15

d = max(maxx-minx, maxy-miny)
c = 1./d

mi = int(len(MS[0])/2)
ms = np.ravel(MS[0][mi])
mx = 0.5 * (ms[0] + ms[1])
my = 0.5 * (ms[2] + ms[3])
mz = 0.5 * (ms[4] + ms[5])

n = len(RS)
MD = [None] * n
MI = [None] * n
for i in range(n):
  rs = RS[i]
  MD[i] = gl.MeshData.sphere(rows=10, cols=20, radius = 0.01)
  col = np.array([1,0,1,1], dtype = float)
  Col = np.tile(col.reshape((1, 4)), (MD[i].faceCount(), 1))
  MD[i].setFaceColors(Col)
  MI[i] = gl.GLMeshItem(meshdata=MD[i], smooth=False, shader = "shaded")
  MI[i].translate((rs[0]-mx)*c, (rs[1]-my)*c, (rs[2]-mz)*c)
  Vbox.addItem(MI[i])


'''
md = gl.MeshData.sphere(rows=4, cols=8)
m4 = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=False, drawEdges=True, edgeColor=(1,1,1,1))
m4.translate(0,10,0)
w.addItem(m4)
'''

TC = [[]] * len(MS)

for h in range(len(MS)):
  TC[h] = [[]] * len(MS[h])
  for i in range(len(MS[h])):
    k = 0
    ms = np.ravel(MS[h][i])
    n = len(ms)/15
    TC[h][i] = [[]] * n
    dd = np.ravel(DD[h][i])
    g = -1
    for j in range(n):
      mshij = ms[k:k+15]
      xyz0 = ((mshij[0]-mx)*c, (mshij[2]-my)*c, (mshij[4]-mz)*c)
      xyz1 = ((mshij[1]-mx)*c, (mshij[3]-my)*c, (mshij[5]-mz)*c)
      rr = (mshij[6]*c, mshij[7]*c)
      g += 1
      rgb = jetrgb(dd[g] / maxdd)
      TC[h][i][j] = truncone(xyz0, xyz1, rr, shader="shaded", color = (rgb[0], rgb[1], rgb[2], 1.))
      Vbox.addItem(TC[h][i][j])
      k += 15


nframes = 360
Anim = anim(Vbox)
Anim.setAnimFunc(Vbox.orbit, 360./nframes, 360./nframes)
Anim.animate(nframes)
Anim.saveAnim()

