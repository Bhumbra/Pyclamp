XLRDVERSION = 0.
USEOPENPYXL = False
NOTANUMBER = -1.7976931348623157e308

try:
  import cpickle as pickl
except ImportError:
  import pickle as pickl

try:
  import odf.opendocument as opendocument
  from odf.table import Table as opendocumentTable
  from odf.table import TableRow as opendocumentTableRow
  from odf.table import TableCell as opendocumentTableCell
  from odf.text import P as opendocumenttextP
except:
  opendocument = None

try:
  import xlrd
  XLRDVERSION = float(xlrd.__VERSION__[:3])
except ImportError:
  pass

if XLRDVERSION < 0.8:
  try:
    import openpyxl
    USEOPENPYXL = True
  except ImportError:
    pass

import csv
import os
import fnmatch
import numpy as np
from pyclamp.dsp.fpfunc import nan2val
from pyclamp.dsp.strfunc import str2rci
from pyclamp.dsp.lsfunc import *

def mkdir(dirn):
  if not os.path.exists(dirn):
    if not os.path.exists(os.path.abspath('') + dirn):
      if not os.path.exists(os.path.abspath('') + '/' + dirn):
        os.mkdir(dirn)

def lsdir(dn = None, spec = [], opts = 0): # opts = 0 files, 1 = files+stems, 2 = files+stems+extns
  if dn is None: dn = os.getcwd()
  if type(spec) is str:
    if len(spec): spec = [spec]
  fn = os.listdir(dn)
  n = len(fn)
  N = len(spec)
  if not(N):
    if not(opts): return fn
    stem = [''] * n
    extn = [''] * n
    for i in range(n):
      stem[i], extn[i] = os.path.splitext(fn[i])
    if opts == 1: return fn, stem
    return fn, stem, extn
  filn = []
  stem = []
  extn = []
  for i in range(n):
    ste, ext = os.path.splitext(fn[i])
    addfn= False
    for j in range(N):
      sp = spec[j]
      isextn = False
      if len(sp):
        if sp[0] == '.': isextn = True
      if not(addfn):
        addfn = ext.lower() == sp.lower() if isextn else (len(fnmatch.filter([fn[i]], sp)) > 0)
    if addfn:
      filn.append(fn[i])
      stem.append(ste)
      extn.append(ext)
  if not(opts): return filn
  if opts == 1: return filn, stem
  return filn, stem, extn

def readDTFile(fn, del0 = '\t', del2 = '\f'):
  f = open(fn, 'rt')
  reader = csv.reader(f, delimiter = del0)
  rows = []
  c = 0
  i = [c]
  for row in reader:
    c += 1
    rows.append(row)
    if len(row) == 1 and del2 is not None:
      if row[0] == del2:
        i.append(c)
  f.close()
  n = len(i)
  Data = [[]] * n
  for j in range(n):
    k = len(rows) if j == n - 1 else i[j+1]-1
    data = rows[i[j]:k]
    Data[j] = data[:]
    single = len(Data[j]) > 1
    for k in range(len(Data[j])): # convention of tuples indicating no horizontal delimiters
      if len(Data[j][k]) == 1:
        data[k] = Data[j][k][0]
      else:
        single = False
    if single:
      Data[j] = tuple(data)
  return Data

def readDTData(fn, del0 = '\t', del2 = '\f'):
  if type(fn) is str:
    data = readDTFile(fn, del0, del2)
  else:
    data = fn
  Data = [[]] * len(data)
  j = 0
  for i in range(len(data)):
    if len(data[i]):
      Data[j] = list2array(list(data[i])) # tuple -> list -> array (if vector)
      j += 1
  return Data[:j]

def writeDTFile(fn, _data, del0 = None, del2 = '\f'):
  if del0 is None: del0 = '\t'
  if type(_data) is list:
    data = _data[:]
  elif isarray(_data):
    data = list(_data)
  else:
    data = [_data]
  n = len(data)
  for i in range(n):
    if type(data[i]) is str:
      data[i] = [data[i]]
    elif type(data[i]) is not tuple:
      if isarray(data[i]):
        data[i] = list(data[i])
      else:
        data[i] = [data[i]]
  f = open(fn, 'wt')
  for i in range(n):
    di = data[i]
    if type(di) is tuple: # tuple arrays are not horizontally delimited
      Di = list(di)
      for di in Di:
        f.write(str(di))
        f.write('\n')
    else:
      ndi = len(di)
      for j in range(ndi):
        dj = di[j]
        if type(dj) is str:
          f.write(dj)
          f.write('\n')
        elif isnum(dj):
          f.write(str(dj))
          f.write('\n')
        else:
          writer = csv.writer(f, delimiter = del0)  # writes bullshit
          writer.writerow(dj)                       # brackets
    if i < n - 1 and del2 is not None:
      writer = csv.writer(f, delimiter = del0)
      writer.writerow(del2)
  f.close()

def readODFile(fn, sn = 0):
  if opendocument is None: raise ImportError("No module named odf could be imported")
  od = opendocument.load(fn)
  ods = od.spreadsheet.getElementsByType(opendocumentTable)
  rows = ods[sn].getElementsByType(opendocumentTableRow)
  data = []
  ncols = 0
  for row in rows:
    cells = row.getElementsByType(opendocumentTableCell)
    ncols = max(len(cells), ncols)
  for row in rows:
    cells = row.getElementsByType(opendocumentTableCell)
    rowData = [None] * ncols
    haveData = False
    j = 0
    for cell in cells:
      nc = cell.getAttribute("numbercolumnsrepeated") # what a rubbish format!
      nc = 1 if (not nc) else int(nc)
      element = cell.getElementsByType(opendocumenttextP)
      celldata = ""
      for el in element:
        for cn in el.childNodes:
          if cn.nodeType == 3:
            celldata += unicode(cn.data)
      if len(celldata):
        cellData= float(celldata) if isnumeric(celldata) else celldata
        haveData = True
        for k in range(j, j+nc):
          rowData[k] = cellData
      j += nc
    if haveData: data.append(rowData)
  return data

def readODData(fn, sn = 0, ret1stRow = False):
  data = readODFile(fn, sn)
  nrows = len(data)
  ncols = len(data[0]) if nrows else 0
  Data = np.tile( (np.array(np.NaN)), (ncols, nrows))
  for j in range(ncols):
    for i in range(nrows):
      d = data[i][j]
      if type(d) is float or type(d) is int:
        Data[j][i] = d
  notnancols = np.nonzero(np.sum(np.isnan(Data), axis=1) < nrows)[0]
  DataOK = Data[notnancols]
  if ret1stRow: return DataOK, data[0]
  return DataOK

def readXLFile(fn, sn = 0):
  if not(USEOPENPYXL) or os.path.splitext(fn)[1].lower() == '.xls':
    book = xlrd.open_workbook(fn)
    sheet = book.sheet_by_index(sn)
    data = [[]] * sheet.nrows
    for i in range(sheet.nrows):
      data[i] = sheet.row_values(i)
    return data
  book = openpyxl.load_workbook(fn)
  sheet = book.get_sheet_by_name(name = book.get_sheet_names()[sn])
  data = [[]] * len(sheet.rows)
  for i in range(len(sheet.rows)):
    data[i] = [[]] * len(sheet.rows[i])
    for j in range(len(sheet.rows[i])):
      data[i][j] = sheet.rows[i][j].value
  return data

def readXLData(fn, sn = 0, ret1stRow = False):
  data = readXLFile(fn, sn)
  nrows = len(data)
  ncols = len(data[0]) if nrows else 0
  Data = np.tile( (np.array(np.NaN)), (ncols, nrows))
  for j in range(ncols):
    for i in range(nrows):
      d = data[i][j]
      if type(d) is float or type(d) is int:
        Data[j][i] = d
  notnancols = np.nonzero(np.sum(np.isnan(Data), axis=1) < nrows)[0]
  DataOK = Data[notnancols]
  if ret1stRow: return DataOK, data[0]
  return DataOK

def readDTMapData(fn, lbl = []):
  Data = readDTFile(fn)
  N = len(Data)
  M = len(lbl)
  X = [[]] * M
  for i in range(M):
    X[i] = np.empty(N, dtype = float)
  for i in range(N):
    data = Data[i]
    for j in range(len(data)):
      for k in range(M):
        if data[j][0] == lbl[k]:
          X[k][i] = float(data[j][1]);
  return X;

def readBinFile(fn, datatype = np.float64):
  f = open(fn, 'rb')
  Data = np.fromfile(file=f, dtype = datatype)
  f.close();
  return Data

def writeBinFile(fn = None, _Data = None):
  if type(_Data) is np.ndarray:
    Data = np.copy(_Data)
  elif type(_Data) is list or type(_Data) is tuple:
    Data = _Data[:]
  else:
    Data = _Data
  f = open(fn, 'wb') if type(fn) is str else fn
  if (type(Data) is list or type(Data) is tuple): # recursion for lists
    for data in Data:
      writeBinFile(f, data)
    return
  f.write(Data)
  if type(fn) is str: f.close()

def readSCFile(fn, nanconv = False):
  data = readDTFile(fn)[0]
  n = len(data)
  ei = np.tile(-1, n)
  st = np.tile(False, n)
  nm = np.tile(False, n)
  I = np.tile(-1, n)
  J = np.tile(-1, n)
  for i in range(n):
    if len(data[i]):
      S = data[i][0]
      ei[i] = S.find('=')
    if ei[i] > 0:
      s = S[:ei[i]-1]
      istr = s.find('string')
      ilet = s.find('let')
      st[i] = istr >= 0
      nm[i] = ilet >= 0
      irc = None
      if st[i]:
        irc = istr + 7
      elif nm[i]:
        irc = ilet + 4
      if irc is not None:
        sirc = s[irc:]
        _I, _J = str2rci(sirc)
        if _I >= 0 and _J >= 0:
          I[i], J[i] = _I, _J
  nr = I.max()+1
  nc = J.max()+1
  ok = np.logical_and(I >= 0, J >= 0)
  Data = [[]] * nr
  for i in range(nr): Data[i] = [""] * nc
  for i in range(n):
    if ok[i]:
      r, c = I[i], J[i]
      if st[i]:
        Data[r][c] = data[i][0][ei[i]+2:].replace('"', '')
        if nanconv: Data[r][c] = np.nan
      elif nm[i]:
        Data[r][c] = float(data[i][0][ei[i]+2:])
  return Data

def readSVFile(fn, delim = None, nanconv = True):
  if delim is None: delim = '\t'
  if type(fn is str):
    data = flattenarray(readDTFile(fn, delim, None)[0])
  else:
    data = fn
  N = len(data)
  if N < 4: return data
  k = 0
  n = int(data[k])
  if n != 3:
    raise ImportError("Cannot import with non-three dimensionality")
  k += 1
  n = int(data[k])
  X = [None] * n
  k += 1
  for h in range(n):
    nr = int(data[k])
    k += 1
    x = [None] * nr
    Nc = np.empty(nr, dtype = int)
    NC = np.tile(nanconv, nr)
    for i in range(nr):
      nc = int(data[k])
      Nc[i] = nc
      k += 1
      l = k + nc
      xi = data[k:l]
      isN, isn = isNumeric(xi, True)
      if isN:
        xi = np.array(xi, dtype = float)
        xi[xi == NOTANUMBER] = np.NaN
      else:
        NC[i] = False
        for j in range(len(isn)):
          if isn[j]: xi[j] = float(xi[j])
      x[i] = xi
      k = l
      if h + 1 < n and k >= N:
        print("Warning: dimension specification exceeds file size")
        return X
    X[h] = np.array(x) if Nc.min() == Nc.max() and np.all(NC) else x
  if k < N: print("Warning: dimension specification below file size")
  return X

def writeSVFile(fn, _X, delim = None, writeDims = True, nanconv = True):
  if delim is None: delim = '\t'
  ndim = 3
  nd = ndim - 1
  X = _X if isarray(_X) else [_X]
  ns = len(X)
  data = [ndim, ns] if writeDims else [] # first datum shows dimensionality
  for h in range(ns):
    x = X[h]
    if nDim(x) > nd:
      raise ValueError('Data dimensionality larger than maximum of three dimensions')
    else:
      if not(isarray(x)):
        x = [x]
      elif type(x) is tuple:
        x = list(x)
      while nDim(x) < nd:
        x = [x]
    nr = len(x) # number of rows
    if writeDims: data.append(nr)
    for i in range(nr): # iterating across rows
      r = x[i]
      if not(isarray(r)): r = [r]
      nc = len(r)
      if writeDims: data.append(nc)
      if nanconv: r = nan2val(r, NOTANUMBER)
      data += array2list(r)
  writeDTFile(fn, [data], delim, None)
  return data

def readPickle(fn, *args):
  fh = open(fn, 'rb')
  data = pickl.load(fh, *args)
  fh.close()
  return data

def writePickle(fn, data, *args):
  fh = open(fn, 'wb')
  pickl.dump(data, fh, *args)
  fh.close()
  return fh

