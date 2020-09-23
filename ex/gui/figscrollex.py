import wx
import matplotlib as mpl
mpl.use('WXAgg')
import matplotlib.pyplot as mp
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import arange, sin, pi

class ScrollbarFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Scrollbar example',size=(300,200))
        print("self")
        print(self)
        self.scroll = wx.ScrolledWindow(self, -1)
        print("self.scroll")
        print(self.scroll)
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.axes = self.figure.add_subplot(2,1,1)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        self.axes.plot(t,s)
        self.canvas = FigureCanvas(self.figure) #(self, -1, self.figure)
        self.scroll.SetScrollbars(5,5,600,400)
        #self.scroll.FitInside()
   
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame =  ScrollbarFrame()
    mp.subplot(2,1,2)
    frame.Show()
    mp.show()
    app.MainLoop()


