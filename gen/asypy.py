import pyasy.asymptote
from iofunc import *

class asypy:
  asy = None
  picfunc = ["draw", "dot", "label"]
  def __init__(self):
    self.astr = []
    self.nstr = 0
    self.npre = 0
    self.npos = 0
    self.pic()
  def pic(self, _pic = None, position = (0,0), align = "NE"):
    self.pstr = _pic
    if self.pstr is None: return
    self.add("picture " + self.pstr + ";")
    self.pos("add(" + self.pstr + ".fit(), position = " + str(position) + ", align = " + align + ");")
  def pre(self, _str = ''):
    if _str == '': return
    if _str[-1] != ";": _str += ";"
    if self.npre:
      if listfind(self.astr[:self.npre], _str) >= 0: return
    self.astr.insert(self.npre, _str)
    self.npre += 1
    self.nstr += 1
  def add(self, _str = ''):
    if _str == '': return
    if _str[-1] != ";": _str += ";"
    _str = self.apic(_str) 
    self.astr.insert(self.nstr - self.npos,_str)
    self.nstr += 1
  def pos(self, _str = ''):
    if _str == '': return
    if _str[-1] != ";": _str += ";"
    if self.npos:
      if listfind(self.astr[-self.npos:], _str) >= 0: return
    self.astr.append(_str)
    self.nstr += 1
    self.npos += 1
  def apic(self, _str = ""):
    if self.pstr is None: return _str
    if _str.find(self.pstr) >= 0: return _str
    for picfun in self.picfunc:
      ifun = _str.find(picfun)
      if ifun >= 0:
        stri = _str[ifun:]
        if stri.find(picfun+"()") >= 0: # no inputs
          _str = _str.replace(picfun+"(", picfun+"("+self.pstr)
        else:
          _str = _str.replace(picfun+"(", picfun+"("+self.pstr+", ")
    return _str      
  def asy(self, output = None):
    if output is None:
      raise ValueError("Output filestem mandatory")
    if not(self.nstr): return
    if len(output) > 3:
      if output.lower()[-4:] != ".asy":
        output += ".asy"
    else:
      output += ".asy"
    writeDTFile(output, self.astr, '\t', None)
  def eps(self, output = None):
    if output is None:
      raise ValueError("Output filestem mandatory")
    if not(self.nstr): return
    self.asy = pyasy.asymptote.Asymptote()
    for i in range(self.nstr):
      self.asy.send(self.astr[i])
    self.asy.shipout('shipout("' + output + '");') 

ASY = asypy()

def jetstr(num, dem = 0):
  f = float(num)
  if dem > 0:
    f /= float(dem)
  return ''.join(('jet(', str(f), ')'))

