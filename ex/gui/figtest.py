# Todd Miller jmiller@stsci.edu 

from __future__ import division 

import os, sys, math 

import Tkinter as Tk, FileDialog 
import tkagg # Paint image to Tk photo blitter extension 
from backend_agg import FigureCanvasAgg 
 
import os.path 
 
import matplotlib 
from matplotlib.cbook import is_string_like, enumerate, True, False 
from matplotlib.backend_bases import RendererBase, GraphicsContextBase, FigureManagerBase, FigureCanvasBase, NavigationToolbar2, cursors, MplEvent 

from matplotlib.figure import Figure 
from matplotlib._matlab_helpers import Gcf 
from matplotlib.numerix import asarray 

import matplotlib.windowing as windowing 

rcParams = matplotlib.rcParams 
verbose = matplotlib.verbose 


backend_version = Tk.TkVersion 

# the true dots per inch on the screen; should be display dependent 
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi 
PIXELS_PER_INCH = 75 
 
cursord = { 
cursors.MOVE: "fleur", 
cursors.HAND: "hand2", 
cursors.POINTER: "arrow", 
cursors.SELECT_REGION: "tcross", 
} 

def round(x): 
 return int(math.floor(x+0.5)) 

def raise_msg_to_str(msg): 
 """msg is a return arg from a raise. Join with new lines""" 
 if not is_string_like(msg): 
 msg = '\n'.join(map(str, msg)) 
 return msg 

def error_msg_tkpaint(msg, parent=None): 
 import tkMessageBox 
 tkMessageBox.showerror("matplotlib", msg) 

def draw_if_interactive(): 
 if matplotlib.is_interactive(): 
 figManager = Gcf.get_active() 
 if figManager is not None: 
 figManager.show() 

def show(): 
 """ 
 Show all the figures and enter the gtk mainloop 

 This should be the last line of your script 
 """ 
 for manager in Gcf.get_all_fig_managers(): 
 manager.show() 
 import matplotlib 
 matplotlib.interactive(True) 
 #os.environ['PYTHONINSPECT'] = '1' 
 if show._needmain: 
 Tk.mainloop() 
 show._needmain = False 
 show._needmain = True 

def new_figure_manager(num, *args, **kwargs): 
 """ 
 Create a new figure manager instance 
 """ 
 _focus = windowing.FocusManager() 
 figure = Figure(*args, **kwargs) 
 window = Tk.Tk() 
 canvas = FigureCanvasTkAgg(figure, master=window) 
 figManager = FigureManagerTkAgg(canvas, num, window) 
 return figManager 


class FigureCanvasTkAgg(FigureCanvasAgg): 
 keyvald = {65507 : 'control', 
 65505 : 'shift', 
 65513 : 'alt', 
 65508 : 'control', 
 65506 : 'shift', 
 65514 : 'alt', 
 } 
 
def __init__(self, figure, master=None, resize_callback=None): 
 FigureCanvasAgg.__init__(self, figure) 
 t1,t2,w,h = self.figure.bbox.get_bounds() 
 w, h = int(w), int(h) 
 self._tkcanvas = Tk.Canvas( 
 master=master, width=w, height=h, borderwidth=4) 
 self._tkphoto = Tk.PhotoImage( 
 master=self._tkcanvas, width=w, height=h) 
 self._tkcanvas.create_image(w/2, h/2, image=self._tkphoto) 
 self._resize_callback = resize_callback 
 self._tkcanvas.bind("<Configure>", self.resize) 
 self._tkcanvas.bind("<Key>", self.key_press) 
 self._tkcanvas.bind("<KeyRelease>", self.key_release) 
 self._master = master 
 self._tkcanvas.focus_set() 
 self._key = None # the key that is pressed  

def resize(self, event): 
 width, height = event.width, event.height 
 if self._resize_callback is not None: 
 self._resize_callback(event) 

 # compute desired figure size in inches 
 dpival = self.figure.dpi.get() 
 winch = width/dpival 
 hinch = height/dpival 
 self.figure.set_figsize_inches(winch, hinch) 


 self._tkphoto.configure(width=width, height=height) 
 self._tkcanvas.delete(self._tkphoto) 
 self._tkphoto = Tk.PhotoImage( 
 master=self._tkcanvas, width=width, height=height) 
 self._tkcanvas.create_image(width/2,height/2,image=self._tkphoto) 
 self.show() 

def draw(self): 
 FigureCanvasAgg.draw(self) 
 tkagg.blit(self._tkphoto, self.renderer._renderer, 2) 
 self._master.update_idletasks() 

 show = draw 

def get_tk_widget(self): 
 """returns the Tk widget used to implement FigureCanvasTkAgg. 
 Although the initial implementation uses a Tk canvas, this routine 
 is intended to hide that fact. 
 """ 
 return self._tkcanvas 

def print_figure(self, filename, dpi=150, 
 facecolor='w', edgecolor='w', 
 orientation='portrait'): 

 agg = self.switch_backends(FigureCanvasAgg) 
 agg.print_figure(filename, dpi, facecolor, edgecolor, orientation) 

def connect(self, eventname, handler): 
 tkname = { 
 "motion_notify_event":["<Motion>"], 
 "button_press_event": ["<Button-1>", 
 "<Button-2>", 
 "<Button-3>", 
 ], 
 "button_release_event": ["<ButtonRelease-1>", 
 "<ButtonRelease-2>", 
 "<ButtonRelease-3>", 
 ], 
 "key_press_event":["<Key>"], 
 "key_release_event":["<KeyRelease>"], 
 }
 
def the_binding(e): 
 e.button = e.num 
 return handler(e.widget, e) 
 for name in tkname[eventname]: 
  self._tkcanvas.bind(name, the_binding) 
 return (eventname, the_binding) 

def key_press(self, event): 
 val = event.keysym_num 
 if self.keyvald.has_key(val): 
 key = self.keyvald[val] 
 elif val<256: 
 key = chr(val) 
 else: 
 key = None 

 self._key = key 

def key_release(self, event): 
 self._key = None 


def mpl_connect(self, s, func): 
 if s not in self.events: 
 error_msg('Can only connect events of type "%s"\nDo not know how to handle "%s"' %(', '.join(self.events), s)) 

 needPosition = s in ('button_press_event', 
 'button_release_event', 
 'motion_notify_event', 
 )   

 needButton = s in ('button_press_event', 
 'button_release_event', 
 ) 

def wrapper(widget, event): 
 thisEvent = MplEvent(s, self) 
 if needPosition: 
 thisEvent.x = event.x 
 # flipy so y=0 is bottom of canvas 
 thisEvent.y = self.figure.bbox.height() - event.y  

 #print 'wrapper', event.keysym, event.keycode, event.keysym_num 

 try: num = event.num 
 except AttributeError: thisEvent.button = None 
 else: 
  if num in (1,2,3): 
  if sys.platform=='darwin': 
# 2 and 3 were reversed on the OSX platform I 
# tested under tkagg 
  if num==2: num=3 
  elif num==3: num=2 
thisEvent.button = num 
else: thisEvent.button = None 
thisEvent.inaxes = None 
thisEvent.key = self._key 
for a in self.figure.get_axes(): 
if a.in_axes(thisEvent.x, thisEvent.y): 
thisEvent.inaxes = a 
xdata, ydata = a.transData.inverse_xy_tup((thisEvent.x, thisEvent.y)) 
thisEvent.xdata = xdata 
thisEvent.ydata = ydata 
break 

func(thisEvent) 

return self.connect(s, wrapper) 

def mpl_disconnect(self, cid): 
eventname, the_binding = cid 
self._tkcanvas.unbind(eventname) #, the_binding) 
return None 

class FigureManagerTkAgg(FigureManagerBase): 
""" 
Public attributes 

canvas : The FigureCanvas instance 
num : The Figure number 
toolbar : The tk.Toolbar 
window : The tk.Window 
""" 
def __init__(self, canvas, num, window): 
FigureManagerBase.__init__(self, canvas, num) 
self.window = window 
self.window.withdraw() 
self.window.wm_title("Figure %d" % num) 
self.canvas = canvas 
self._num = num 
t1,t2,w,h = canvas.figure.bbox.get_bounds() 
w, h = int(w), int(h) 
self.window.minsize(int(w*3/4),int(h*3/4)) 
if matplotlib.rcParams['toolbar']=='classic': 
self.toolbar = NavigationToolbar( canvas, self ) 
elif matplotlib.rcParams['toolbar']=='toolbar2': 
self.toolbar = NavigationToolbar2TkAgg( canvas, self ) 
else: 
self.toolbar = None 
if self.toolbar is not None: 
self.toolbar.update() 
self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1) 

def destroy(*args): 
self.window = None 
Gcf.destroy(num) 
self.window.bind("<Destroy>", destroy) 

def resize(self, event): 
width, height = event.width, event.height 
self.toolbar.configure(width=width) # , height=height) 

def show(self): 
_focus = windowing.FocusManager() 
self.window.deiconify() 
self.canvas.draw() 

def add_subplot(self, *args, **kwargs): 
a = FigureManagerBase.add_subplot(self, *args, **kwargs) 
if self.toolbar is not None: 
self.toolbar.update() 
return a 

def add_axes(self, rect, **kwargs): 
a = FigureManagerBase.add_axes(self, rect, **kwargs) 
if self.toolbar is not None: 
self.toolbar.update() 
return a 

def set_current_axes(self, a): 
if a not in self.axes.values(): 
error_msg_tkpaint('Axes is not in current figure') 
FigureManagerBase.set_current_axes(self, a) 

def destroy(self, *args): 
if Gcf.get_num_fig_managers()==0 and not matplotlib.is_interactive(): 
if self.window is not None: 
self.window.quit() 
if self.window is not None: 
self.window.destroy() 
self.window = None 

class AxisMenu: 
def __init__(self, master, naxes): 
self._master = master 
self._naxes = naxes 
self._mbar = Tk.Frame(master=master, relief=Tk.RAISED, borderwidth=2) 
self._mbar.pack(side=Tk.LEFT) 
self._mbutton = Tk.Menubutton( 
master=self._mbar, text="Axes", underline=0) 
self._mbutton.pack(side=Tk.LEFT, padx="2m") 
self._mbutton.menu = Tk.Menu(self._mbutton) 
self._mbutton.menu.add_command( 
label="Select All", command=self.select_all) 
self._mbutton.menu.add_command( 
label="Invert All", command=self.invert_all) 
self._axis_var = [] 
self._checkbutton = [] 
for i in range(naxes): 
self._axis_var.append(Tk.IntVar()) 
self._axis_var[i].set(1) 
self._checkbutton.append(self._mbutton.menu.add_checkbutton( 
label = "Axis %d" % (i+1), 
variable=self._axis_var[i], 
command=self.set_active)) 
self._mbutton.menu.invoke(self._mbutton.menu.index("Select All")) 
self._mbutton['menu'] = self._mbutton.menu 
self._mbar.tk_menuBar(self._mbutton) 
self.set_active() 

def adjust(self, naxes): 
if self._naxes < naxes: 
for i in range(self._naxes, naxes): 
self._axis_var.append(Tk.IntVar()) 
self._axis_var[i].set(1) 
self._checkbutton.append( self._mbutton.menu.add_checkbutton( 
label = "Axis %d" % (i+1), 
variable=self._axis_var[i], 
command=self.set_active)) 
elif self._naxes > naxes: 
for i in range(self._naxes-1, naxes-1, -1): 
del self._axis_var[i] 
self._mbutton.menu.forget(self._checkbutton[i]) 
del self._checkbutton[i] 
self._naxes = naxes 
self.set_active() 

def get_indices(self): 
a = [i for i in range(len(self._axis_var)) if self._axis_var[i].get()] 
return a 

def set_active(self): 
self._master.set_active(self.get_indices()) 

def invert_all(self): 
for a in self._axis_var: 
a.set(not a.get()) 
self.set_active() 

def select_all(self): 
for a in self._axis_var: 
a.set(1) 
self.set_active() 

class NavigationToolbar(Tk.Frame): 
""" 
Public attriubutes 

canvas - the FigureCanvas (gtk.DrawingArea) 
win - the gtk.Window 

""" 
def _Button(self, text, file, command): 
file = os.path.join(rcParams['datapath'], file) 
im = Tk.PhotoImage(master=self, file=file) 
b = Tk.Button( 
master=self, text=text, padx=2, pady=2, image=im, command=command) 
b._ntimage = im 
b.pack(side=Tk.LEFT) 
return b 

def __init__(self, canvas, figman): 
xmin, xmax = canvas.figure.bbox.intervalx().get_bounds() 
height, width = 50, xmax-xmin 
Tk.Frame.__init__(self, master=figman.window, 
width=width, height=height, 
borderwidth=2) 
self.canvas = canvas 
self.figman = figman 

self.update() # Make axes menu 

self.bLeft = self._Button( 
text="Left", file="stock_left.ppm", 
command=lambda x=-1: self.panx(x)) 

self.bRight = self._Button( 
text="Right", file="stock_right.ppm", 
command=lambda x=1: self.panx(x)) 

self.bZoomInX = self._Button( 
text="ZoomInX",file="stock_zoom-in.ppm", 
command=lambda x=1: self.zoomx(x)) 

self.bZoomOutX = self._Button( 
text="ZoomOutX", file="stock_zoom-out.ppm", 
command=lambda x=-1: self.zoomx(x)) 

self.bUp = self._Button( 
text="Up", file="stock_up.ppm", 
command=lambda y=1: self.pany(y)) 

self.bDown = self._Button( 
text="Down", file="stock_down.ppm", 
command=lambda y=-1: self.pany(y)) 

self.bZoomInY = self._Button( 
text="ZoomInY", file="stock_zoom-in.ppm", 
command=lambda y=1: self.zoomy(y)) 

self.bZoomOutY = self._Button( 
text="ZoomOutY",file="stock_zoom-out.ppm", 
command=lambda y=-1: self.zoomy(y)) 

self.bSave = self._Button( 
text="Save", file="stock_save_as.ppm", 
command=self.save_figure) 

self.pack(side=Tk.BOTTOM, fill=Tk.X) 

def close(self): 
Gcf.destroy(self.figman._num) 

def set_active(self, ind): 
self._ind = ind 
self._active = [ self._axes[i] for i in self._ind ] 

def panx(self, arg): 
try: arg.direction 
except AttributeError: direction = arg 
else: 
if arg.direction == Tk.SCROLL_UP: direction=1 
else: direction=-1 
for a in self._active: 
a.panx(direction) 
self.figman.show() 

def pany(self, arg): 
try: arg.direction 
except AttributeError: direction = arg 
else: 
if arg.direction == Tk.SCROLL_UP: direction=1 
else: direction=-1 
for a in self._active: 
a.pany(direction) 
self.figman.show() 

def zoomx(self, arg): 
try: arg.direction 
except AttributeError: direction = arg 
else: 
if arg.direction == Tk.SCROLL_UP: direction=1 
else: direction=-1 

for a in self._active: 
a.zoomx(direction) 
self.figman.show() 

def zoomy(self, arg): 
try: arg.direction 
except AttributeError: direction = arg 
else: 
if arg.direction == Tk.SCROLL_UP: direction=1 
else: direction=-1 

for a in self._active: 
a.zoomy(direction) 
self.figman.show() 

def save_figure(self): 
fs = FileDialog.SaveFileDialog(master=self.figman.window, 
title='Save the figure') 
try: 
self.lastDir 
except AttributeError: 
self.lastDir = os.curdir 

fname = fs.go(dir_or_file=self.lastDir) # , pattern="*.png") 
if fname is None: # Cancel 
return 

self.lastDir = os.path.dirname(fname) 
try: 
self.canvas.print_figure(fname) 
except IOError, msg: 
err = '\n'.join(map(str, msg)) 
msg = 'Failed to save %s: Error msg was\n\n%s' % ( 
fname, err) 
error_msg_tkpaint(msg) 

def update(self): 
_focus = windowing.FocusManager() 
self._axes = self.canvas.figure.axes 
naxes = len(self._axes) 
if not hasattr(self, "omenu"): 
self.set_active(range(naxes)) 
self.omenu = AxisMenu(master=self, naxes=naxes) 
else: 
self.omenu.adjust(naxes) 

class NavigationToolbar2TkAgg(NavigationToolbar2, Tk.Frame): 
""" 
Public attriubutes 

canvas - the FigureCanvas (gtk.DrawingArea) 
win - the gtk.Window 
""" 
def __init__(self, canvas, figman): 
self.canvas = canvas 
self.figman = figman 
self._idle = True 
Tk.Frame.__init__(self, master=self.canvas._tkcanvas) 
NavigationToolbar2.__init__(self, canvas) 

def set_message(self, s): 
self.message.set(s) 

def draw_rubberband(self, event, x0, y0, x1, y1): 
height = self.canvas.figure.bbox.height() 
y0 = height-y0 
y1 = height-y1 
try: self.lastrect 
except AttributeError: pass 
else: self.canvas._tkcanvas.delete(self.lastrect) 
self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1) 

#self.canvas.draw() 

def release(self, event): 
try: self.lastrect 
except AttributeError: pass 
else: 
self.canvas._tkcanvas.delete(self.lastrect) 
del self.lastrect 

def set_cursor(self, cursor): 
self.figman.window.configure(cursor=cursord[cursor]) 

def _Button(self, text, file, command): 
file = os.path.join(rcParams['datapath'], file) 
im = Tk.PhotoImage(master=self, file=file) 
b = Tk.Button( 
master=self, text=text, padx=2, pady=2, image=im, command=command) 
b._ntimage = im 
b.pack(side=Tk.LEFT) 
return b 

def _init_toolbar(self): 
xmin, xmax = self.canvas.figure.bbox.intervalx().get_bounds() 
height, width = 50, xmax-xmin 
Tk.Frame.__init__(self, master=self.figman.window, 
width=width, height=height, 
borderwidth=2) 

self.update() # Make axes menu 

self.bHome = self._Button( text="Home", file="home.ppm", 
command=self.home) 

self.bBack = self._Button( text="Back", file="back.ppm", 
command = self.back) 

self.bForward = self._Button(text="Forward", file="forward.ppm", 
command = self.forward) 

self.bPan = self._Button( text="Pan", file="move.ppm", 
command = self.pan) 

self.bZoom = self._Button( text="Zoom", 
file="zoom_to_rect.ppm", 
command = self.zoom) 

self.bsave = self._Button( text="Save", file="filesave.ppm", 
command = self.save_figure) 
self.message = Tk.StringVar(master=self) 
self._message_label = Tk.Label(master=self, textvariable=self.message) 
self._message_label.pack(side=Tk.RIGHT) 
self.pack(side=Tk.BOTTOM, fill=Tk.X) 

def save_figure(self): 
fs = FileDialog.SaveFileDialog(master=self.figman.window, 
title='Save the figure') 
try: 
self.lastDir 
except AttributeError: 
self.lastDir = os.curdir 

fname = fs.go(dir_or_file=self.lastDir) # , pattern="*.png") 
if fname is None: # Cancel 
return 

self.lastDir = os.path.dirname(fname) 
try: 
self.canvas.print_figure(fname) 
except IOError, msg: 
err = '\n'.join(map(str, msg)) 
msg = 'Failed to save %s: Error msg was\n\n%s' % ( 
fname, err) 
error_msg_tkpaint(msg) 

def set_active(self, ind): 
self._ind = ind 
self._active = [ self._axes[i] for i in self._ind ] 

def update(self): 
_focus = windowing.FocusManager() 
self._axes = self.canvas.figure.axes 
naxes = len(self._axes) 
#if not hasattr(self, "omenu"): 
# self.set_active(range(naxes)) 
# self.omenu = AxisMenu(master=self, naxes=naxes) 
#else: 
# self.omenu.adjust(naxes) 
NavigationToolbar2.update(self) 

def dynamic_update(self): 
'update drawing area only if idle' 
d = self._idle 
self._idle = False 
def idle_draw(*args): 
self.canvas.draw() 
self._idle = True 

if d: self.canvas._tkcanvas.after_idle(idle_draw) 


FigureManager = FigureManagerTkAgg 
error_msg = error_msg_tkpaint 

