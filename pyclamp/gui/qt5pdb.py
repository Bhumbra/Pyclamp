from PyQt5.QtCore import pyqtRemoveInputHook
import sys

def set_trace():
  pyqtRemoveInputHook()
  try:
    import pdb
    debugger = pdb.Pdb()
    debugger.reset()
    debugger.do_next(None)
    frame = sys._getframe().f_back
    debugger.interaction(frame, None)
  finally:
    pass
