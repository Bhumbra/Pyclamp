# Loads a Pyclamp TDF analysis files, instantiates Pyclamp, runs the log, and initialises GUI

from pyclamp import pyclamp, Pyclamp
from iofunc import *
import lbwgui as lbw

# PARAMETERS

tdfdn = "/home/admin/data/abcdtraces/"   # Source tdf directory
abfdn = "/home/admin/data/abcdtraces/"   # Source abf directory
tdffn = "16519004autIN1.tdf"                # Source tdf file
i = 0                                    # Index of analyses session (i.e. the ith log is run)

# INPUT
 
tdfdata = readDTFile(tdfdn+tdffn)        # Read TDF File including all logs

# PROCESSING

# Instantiate PyQt application with a dummy MDI interfance

App = lbw.LBWidget(None, None, None, 'app')
"""
Form = lbw.LBWidget(None, None, None, 'mainform', "Pyclamp")
Area = lbw.LBWidget(None, None, None, 'childarea')
Form.setChild(Area)
"""
Child = lbw.LBWidget(None, None, None, 'mainform', "Data File")

# Start Pyclamp

self = Pyclamp(Child)                    # Instantiate Pyclamp
self.writeDisabled = True                # Disable analyses writes
self.setDataDir(abfdn)                   # Point data source to abf directory
self.runstr(tdfdata[i*2])                # Run the strings of the log
self.writeDisabled = False               # Re-enable analyses writes

# OUTPUT

# Initialise GUI

self.InitGUI()                           # Set up GUI from this point on

