#execfile("iplottest.py")

import matplotlib as mpl
mpl.use('Qt4Agg')
#mpl.rcParams['backend.qt4']='PySide'

from pylab import *
from iplot import *
#ion()
class test:
  def __init__(self):
    figure
    self.aI = isubplot(2, 2, 1)
    plot(linspace(-pi, pi, 100), np.random.rand(100), '.')
    self.ai = iscatter(2, 2, 2)
    #self.ai = isubplot(2, 1, 1)
    n = 400
    x = linspace(-pi, pi, n)
    y = 2.*sin(x)
    c = linspace(0, 1, n)
    C = [[]] * len(x)
    #for i in range(n): C[i] = str(c[i])
    #plot(linspace(-pi, pi, 100), np.random.rand(100), '.', animated = False)
    self.ai.plot(linspace(-pi, pi, 100), np.random.rand(100), '.')
    self.ai.addGUIEllipse()
    
    X = [ ['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'] ]
    self.itp = itxtplot(2, 2, 4)
    self.itp.setData(X, self.xyClick)
    
    na = exes('r', 3, 5)
    but1 = mpl.widgets.Button(na, 'Button1')
    but1.on_clicked(self.click1)    
    na = exes('r', 4, 5)
    but2 = mpl.widgets.Button(na, 'Button2')
    but2.on_clicked(self.click2)       
    na = exes('b', 4, 5, 0.02, -0.04);
    tbtn = mpl.widgets.Button(na, 'Test')
    show()
    
  def click1(self, event = None):
    print("click1")
    self.aI.setCursor([False, True], self.unclick1)
  def unclick1(self, event = None):
    print self.aI.XYGUI;
  def click2(self, event = None):  
    print("click2")
    self.aI.cursorPair([False, True], self.unclick2)
  def unclick2(self, event = None):
    print(self.aI.XYGUI)
  def xyClick(self, event = None):
    i, j = self.itp.event2index(event)
   
iplottest = test()
