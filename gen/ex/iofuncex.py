import csv
import sys
import iofunc
import numpy 

pf = '/home/admin/Test/test.tab'

n = 5

I0 = []
I1 = []
I2 = []

for i in range(n):
  I0.append([i+1, chr(ord('a') + i), '08/%02d/07' % (i+1)])
  I1.append([i+1+n, chr(ord('a') + i+n), '08/%02d/07' % (i+1+n)])
  I2.append([i+1+2*n, chr(ord('a') + i+2*n), '08/%02d/07' % (i+1+2*n)])

I3 = numpy.eye(5)  
  
I = [I0, I1, I2, I3]

#iofunc.writeDTFile(pf, I)

O = iofunc.readDTData(pf)
