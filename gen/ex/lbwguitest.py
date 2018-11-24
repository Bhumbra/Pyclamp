import lbwgui as lbw

lbw.use('qt')

import pylab as pl

def buttonpress0(event = None):
  print("Hello World")
  
class test:
  def __init__(self, Form = None):
    if Form == None: return
    Lbls = [ ["Label 1", "Label 2"], ["Label 3", "Label 4"], ["Label 5", "Label 6" ] ]
    Wids = [ ["edit", "edit"], ["edit", "edit"], ["edit", "listbox"] ]
    Data = [ ["", ""], ["", ""], ["", 0]]
    Opts = [ [None, None], [None, None], [None, ["Opt1", "Opt2"] ]]

    MB = lbw.LBWidget(Form, None, 1)
    BB = lbw.BWidgets(Form, [0, 0], None, ["button", "button"], ["button0", "button1"])
    BB.Widgets[0].connect("btndown", self.buttonpress1)
    
    self.BG = lbw.BGWidgets(Form, 'Button group label', 0, ["Label1", "Label2"], 'radiobutton', 1)    
    
    self.GB = lbw.LBWidgets(Form, 'Label', 1, Lbls, Wids, Data, Opts)
    self.GB.Boxes[2].Widgets[1].setMode(3)
    
    #FW = lbw.LBWidget(Form, None, None, 'figure')
    #fig = FW.retFrame()
    #ax = fig.add_subplot(211)
    #x = pl.linspace(-pl.pi, pl.pi, 400)
    #y = pl.sin(x)
    #ax.plot(x,y)
    
    MB.add(self.GB)
    MB.add(self.BG)
    MB.add(BB)
    
    #MB.add(FW)
    #print self.BG.Widget.checkedButton()
    Form.setChild(MB)
  def buttonpress1(self, event = None):
    print(self.GB.retData())
    print(self.BG.retData())
    

App = lbw.LBWidget(None, None, None, 'app')
Form = lbw.LBWidget(None, None, None, 'dlgform', "title")
testGUI = test(Form)
Form.show()
#fo = Form.dlgFileOpen()
App.loop()
