import os, platform
if platform.platform().startswith("Windows"):
  DISPLAY = "Windows"
else:
  from subprocess import Popen, PIPE
  p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
  p.communicate()
  DISPLAY = None if p.returncode else "DISPLAY"
  if DISPLAY is not None:
    try:
      DISPLAY = os.environ[DISPLAY]
    except KeyError:
      DISPLAY = None      
