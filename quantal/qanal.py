# PARAMETERS

import lbwgui
import qmods
lbwgui.use('qt')

App = lbwgui.LBWidget(None, None, None, 'app')


e = 5.
sres = 128
vres = 64
ares = 1
nres = 32
nmax = 128
ipd = "/home/admin/analyses/"
opd = "/home/admin/results/"

qmods.qmods(e, sres, vres, ares, nres, nmax, ipd, opd)

