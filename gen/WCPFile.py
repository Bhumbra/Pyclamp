#WCP data file format changed to V9.0. Can now support up to 128 channel.
#Header and analysis block sizes now depend upon number of channels.
#Older format WCP files converted to V9 when opened with unchanged copy saved as .bak file.

import struct
import numpy as np
# TODO: datafile is importing WCPfile, not sure it's a good idea to import 
# something from datafile here
from neuropype.datafile import ChannelType 


class WCPFile():
    """File handle: holds path to file and vitalstatistics, 
    including date, time, channel geometry...
    
    WCPFile instances have 3 methods to read data from files:
        readHeader
        readSweepInfo
        readSweepData
    
    all other methods use those three to return data
    """
#################################################################################  
    def __init__(self,pathAndName):
        self.time=0
        #read files recorded with winWCP
        self.fileName=pathAndName
        datafile=open(self.fileName,'rb')
        mn = struct.unpack_from("BBBB", datafile.read(4))
        # Magic to magically extract magic number.
        self.magicNumberString = "0x" + reduce(lambda x, y: x+y, [hex(z)[2:4] for z in list(mn)])
        
        self._cache = {}
        self.policy = {'cache': 'all'}
        datafile.seek(0)
        self.header = self.readHeader(datafile)
        self.HBsize = (int(self.header['NC'] - 1)/8 + 1) * 1024
        self.numSweeps = self.header['NR']
        self._checkEnd(self.numSweeps, datafile)
        datafile.close()
        
 
    
    def readHeader(self, datafile = None):
        '''Return a dictionary with the header data, where 
        
        AD the A/D converter input voltage range (V)
        ADCMAX the Maximum A/D sample value
        CTIME the time at the beginning of the recording (according to the computer)
        DT the A/D sampling interval (s)
        ID the Experiment identification line (user defined during acquisition)
        NBA the No. of 512 byte sectors in a record analysis block
        NBD the No. of 512 byte sectors in a record data block
        NBH the No. of bytes in file header block 
        NC the No. of channels per record.
        NR the No. of records in the file.
        NSVALIGN I DON'T NOW
        NSVCHAN I DON'T NOW
        NSVCURO I DON'T NOW. This looks like a cursor position, so it is probably related to analysis.
        NSVCUR1 I DON'T NOW
        NSVS2P I DON'T NOW
        NSVTYPR I DON'T NOW
        NZ the No. of samples averaged to calculate a zero level.
        TXPERC I DON'T NOW
        VER is WCP version number. (version of the .wcp file, not of the software)
        YGx the Channel x gain factor mV/units
        YNx the Channel x name
        YOx the Channel x offset into sample group in data block
        YRx I HAVE NO IDEA !
        YUx the Channel x units
        YZx the Channel x zero level (A/D bits)
        '''
        if datafile is None:
            datafile=open(self.fileName,'rb')
            close = True
        else:
            close = False
        datafile.seek(0)
        header={}
        while True:
            line = datafile.readline()
            key, value = line.strip().split('=')
            intlist = ['AD','ADCMAX','NBA','NBD','NBH','NC','NR','NZ','VER']
            if key[:2] == 'YG' or key in ['DT', 'TXPERC', 'VER']:
                value = float(value.replace(',','.'))
            elif key[:2] in ['YZ', 'YO', 'YR'] or key in intlist:
                value = int(value)
            header[key] = value
            if key == 'ID':
                break
        if close:
            datafile.close()
        return header
    
    def readSweepInfo(self, index_sweep, datafile = None):
        '''Return a dictionary containing the information from the analysis
        bloc from sweep 'index_sweep'
        
        This dictonary contains
        
        RS the Record status (ACCEPTED, REJECTED)
        RT Record type (TEST, LEAK, etc.)
        GN Group number (Record leak subtraction grouping number)
        t Time recorded (s)
        dt Sampling interval (s)
        MaxVal_x Max positive limit of A/D voltage range (V) of channel x
        mark Marker (User entered marker)
        val Values from waveform measurement and curve fitting data storage array.
        
        For more information refer winWCP user guide
        '''
        
        if datafile is None:
            datafile=open(self.fileName,'rb')
            close = True
        else:
            close = False
        datafile.seek(self.HBsize+ index_sweep*512 * (int(self.header['NBD']) +
                      int(self.header['NBA'])))
        RAB = {}
        RAB['RS'] = datafile.read(8)
        RAB['RT'] = datafile.read(4)
        floats = np.fromfile(datafile, dtype=np.float32,   \
                             count=(3+self.header['NC']),sep="")
        keys = ['GN', 't', 'dt']
        keys += ['MaxVal_'+str(i) for i in range(self.header['NC'])]
        for i, n in enumerate(keys):
            RAB[n] = floats[i]
        RAB['mark'] = datafile.read(16)
        RAB['val'] = np.fromfile(datafile, dtype=np.float32,   \
                             count=((int(self.header['NC']-1)/8+1)*28),sep="")
        if close:
            datafile.close()
        return RAB
    
    def readSweepData(self, index_sweep, datafile = None):
        '''Return the array containing the data stored in the file
        
        Return an array of int16, the actual value, Yfloat for channel x is:
        Yfloat = Yint16 * RAB[MaxVal_x] / (header['YGx']+header[ADCMAX])
        
        The time corresponding to this data is defined by t and dt in RAB'''
        if datafile is None:
            datafile=open(self.fileName,'rb')
            close = True
        else:
            close = False
        
        datafile.seek(self.HBsize + self.header['NBA']*512 +
                      index_sweep*512*(self.header['NBD'] + self.header['NBA']))
        data=np.fromfile(datafile,dtype=np.int16,count=self.header['NBD']*256,sep="")
        data=np.reshape(data,(self.header['NC'],-1),order='F')
                
        if close:
            datafile.close()
        return data
    
    def sweep(self, index_sweep, chan= None, dtype = 'float64'):
        '''Return an 2-D array with time in first row and one row per channel
        
        if dtype is an integer type, data will be given as writen in the file,
        multiply them by mult_factor to convert them in true unit. The time line
        will be kept but not multiplied by dt and is likely to be corrupted (if
        there are more points than the resolution allowed by dtype)'''
        
        if dtype is None:
            dtype = 'float64'
        if chan is None:
            chan = range(self.header['NC'])
        if not isinstance(chan, list):
            chan = [chan]
        datafile=open(self.fileName,'rb')
        RAB = self.readSweepInfo(index_sweep, datafile)
        data = self.readSweepData(index_sweep, datafile)
        datafile.close()
        
        data = data.astype(dtype)
        time = np.arange(data.shape[1], dtype = dtype)
        if dtype not in ['int', 'int16', 'int32', 'int64', 'int8']:
            factor = self.mult_factor(index_sweep, RAB)
            for i in range(self.header['NC']):
                data[i] *= factor[i]
            time *= RAB['dt']
        #time += RAB['t'] #TODO: understand what time recorded is and see what to do with it
        #it's not the length recorded in this peculiar sweep, and doesn't seem to be t[0]
        
        out = np.vstack((time, data[chan]))
        return out
    
    def sweepInfo(self, index_sweep):
        """Return a dict with 'time_recorded' (s), 'dt' (s), 'numChans',
        'numPoints', 't0' (s), 'duration' (s), 'tend' (s), 'absolute_t0',
        and channel info
        
        absolute_t0 is the time when the recording started
        time_recorded is the time recorded since the beginning of the file
        channel info is a list of ChannelType instances containing name, title,
        units, isinput, maxval (V), gain (mV/unit)"""
        NC = self.header['NC']
        NBD = self.header['NBD']
        RAB = self.readSweepInfo(index_sweep)
        out = {'time_recorded': RAB['t'],
               'dt': RAB['dt'],
               'numChans': NC,
               'numPoints': NBD/NC*256,
               't0' : 0,
               'duration': NBD/NC*256*RAB['dt'],
               'tend' : NBD/NC*256*RAB['dt'],
               'absolute_t0' : 'nd'
               }
        channelInfo = [self.channelInfo(index_sweep, i, RAB) for i in range(NC)]
        out['channelInfo'] = channelInfo
 # Return an array of int16, the actual value, Yfloat for channel x is:
 #        Yfloat = Yint16 * RAB[MaxVal_x] / (header['YGx']+header[ADCMAX]
        return out
    
    def samplingInterval(self):
        return self.header['DT']
    
    def channelInfo(self, index_sweep, index_chan, RAB = None):
        
        name = 'Ch_'+str(index_chan)
        units = self.header['YU'+str(index_chan)]
        if RAB is None: RAB = self.readSweepInfo(index_sweep)
        gain = self.gains(index_sweep, RAB)[index_chan]
        maxval = self.maxval(index_sweep, RAB)[index_chan]
        isinput = False
        title = self.header['YN'+str(index_chan)]
        factor = self.mult_factor(index_sweep, RAB)[index_chan]
        return ChannelType(name, title, units, isinput, maxval, factor)
    
    def maxval(self, index_sweep = 0, RAB = None):
        NC = self.header['NC']
        YG = np.asarray([self.header['YG'+str(i)] for i in range(NC)])
        
        if RAB is None:
            RAB = self.readSweepInfo(index_sweep)
        MaxVal = np.asarray([RAB['MaxVal_'+str(i)] for i in range(NC)])
        return MaxVal/YG
    
    def mult_factor(self, index_sweep = 0, RAB = None):
        """return a list of factor such as int16 * factor = true_unit"""
        NC = self.header['NC']
        if RAB is None:
            RAB = self.readSweepInfo(index_sweep)
        
        out = []
        for i in range(NC):
            out.append(RAB['MaxVal_'+str(i)]/(self.header['YG'+str(i)]*self.header['ADCMAX']))
        
        return out
    
    
    def gains(self, index_sweep = 0, RAB =None):
        '''return a list of gain in mV/units'''
        
        if RAB is None:
            RAB = self.readSweepInfo(index_sweep)
        NC = self.header['NC']
        YG = np.asarray([self.header['YG'+str(i)] for i in range(NC)])
        MaxVal = np.asarray([RAB['MaxVal_'+str(i)] for i in range(NC)])
        AD = float(self.header['AD'])
        return AD*YG/MaxVal
    
    def _checkEnd(self, numSweeps, datafile = None):
        """Some winWCP file have a wrong number of file in their header"""
        close = False
        if datafile is None:
            datafile=open(self.fileName,'rb')
            close = True
        datafile.seek(0,2)
        fileLength = datafile.tell()
        n = numSweeps
        foundSweep = False
        while not foundSweep:
            dataBlockEnd = self.HBsize + self.header['NBA']*512 + (n-1)*512*(self.header['NBD'] + self.header['NBA'])+ self.header['NBD']*256
            if dataBlockEnd < fileLength:
                foundSweep = True
            else:
                n -=1
        if n!= numSweeps:
            print "!!! Wrong number of sweeps in file %s"%self.fileName
            print "(%i in file /%i in header) File might be corrupted !!!"%(n,numSweeps)
            self.numSweeps = n
        if close:
            datafile.close()

        #self._sweepInfo= [datafile.read(8)] #ACCEPTED or REJECTED
        #self._sweepInfo+=[datafile.read(4)] #TEST, LEAK, or other name
        #self._sweepInfo+=list(np.fromfile(datafile, dtype=np.float32,   \
        #count=(3+self.header['NC']),sep=""))   
        #datafile.close()
        ##sweep info is [status, type, group number, time of beginning, sampling
        ##interval, max positive A/D voltage for each channel]
        #self.numSweeps = self.header['NR']
        #self._sampleInterval = self.header['DT']
        #self.channelInfo=[]
        #for i in range (self.header['NC']):
            #self.channelInfo.append(ChannelType(name = 'Ch_'+str(i), title =  \
            #self.header['YN'+str(i)], units =self.header['YU'+str(i)], isinput\
            #= False, maxval=self._sweepInfo[5+i], gain=self.header['YG'+str(i)]))
        #self.gains = [i.gain for i in self.channelInfo]
  
    #def set_policy(self, value):
        #if not isinstance(value, dict):
            #raise IOError('policy must be a dictionary')
        #self.policy = value
        #if value.has_key('memmap'):
            #fp = np.memmap('memmap', dtype='float64', mode='w+', shape=(
                 #self.numSweeps, int(self.header['NC'])+1, self.readFromDisk(0)[0].size))
            #for indsweep in range(self.numSweeps):
                #fp[indsweep,:,:] = self.readFromDisk(indsweep)
            #del fp
            #self.memmap = np.memmap('memmap', dtype='float64', mode='r', shape=(
                 #self.numSweeps, int(self.header['NC'])+1, self.readFromDisk(0)[0].size))

    
    #def sweepInfo(self, num_sweep, chan=[]):
        #return self.sweep(num_sweep, chan, 1, 0)
    
    #def scale(self, sweep_index = None):
        ## in  WCP files, scale is : Max positive A/D voltage / gain / ADCMAX
        #if sweep_index is None:
            #return [self._sweepInfo[5+i]/float(self.header['YG'+str(i)])/float(self.header['ADCMAX']) for i in range(int(self.header['NC']))]
        #sweepInfo = self.sweep(index_sweep, info = 1, data = 0)
        #return [sweepInfo[i].maxval/float(self.header['YG'+str(i)])/float(self.header['ADCMAX']) for i in range(int(self.header['NC']))]
    
    #def sweep(self, num_sweep, chan=[],info=0, data=1):
        #if chan == []:
            #chan=range(int(self.header['NC']))
        #elif isinstance(chan,int):
            #chan=[chan]
        #if num_sweep>(int(self.header['NR'])-1):
            #raise ValueError('Error : there are only '+self.header['NR']+' sweeps recorded in the file.')
        #elif any([x>int(self.header['NC']) for x in chan]):
            #raise ValueError('Error : there are only '+self.header['NC']+' channels recorded in the file')
        #trace = None
        #if data:
            #if self.policy.has_key('cache'):
                #cached = self._cache.get(num_sweep)
                #if cached is not None:
                    #trace = self.get_cached(num_sweep, chan)
                    #if not info:
                        #return trace
                    #outinfo = self.readFromDisk(num_sweep, chan, info = 1, data = 0)
                    #return [trace, outinfo]
                #if info:
                    #return self.readFromDisk(num_sweep, chan, info = 1, data = 1)
                #return self.readFromDisk(num_sweep, chan, info = 0, data = 1)
            #elif self.policy.has_key('memmap'):
                #if not info: 
                    #return self.memmap[num_sweep,[0]+[i+1 for i in chan],:]
                #outinfo = self.readFromDisk(num_sweep, chan, info = 1, data = 0)
                #return [self.memmap[num_sweep,[0]+[i+1 for i in chan],:], outinfo]
        #if info:
            #return self.readFromDisk(num_sweep, chan, info = 1, data = 0)
            
    
    #def get_cached(self, num_sweep, chan = []):
        #t0=tme.time()
        #raw_data, dt, scale = self._cache.get(num_sweep)
        #if raw_data is None:
            #raise KeyError('Sweep %s is not in cache!' %num_sweep)
        #trace = raw_data[chan].astype('float64')
        #for i, c in enumerate(chan):
            #trace[i]*= scale[c]
        #time = np.arange(trace.shape[1])*dt
        ##print 'getting cached value'
        #t1 = tme.time() - t0
        #self.time +=t1
        #return np.vstack((time, trace))
    
    #def _remember(self, num_sweep, trace, scale, dt):
        ##print 'remembering'
        #p = self.policy['cache']
        #raw_data = trace.astype('int16')
        #if p == 'all':
            #self._cache[num_sweep] = [raw_data, dt, scale]
        #elif isinstance(p, int):
            #while len(self._cache) >= p:
                #a=self._cache.popitem()
            #self._cache[num_sweep] = [raw_data, dt, scale]

    #def readFromDisk(self, num_sweep, chan=[],info=0, data=1):
        ##print 'Reading from disk !!!'
        ##print 'sweep: %s, chan: %s, info: %s, data: %s'%(num_sweep, chan,info, data)
        ##return an array with the time line 0 and 1 line per channel in chan. If info=1, return a list with 2 objects, the array and the sweepInfo from analysis block
        #if chan == []:
            #chan=range(int(self.header['NC']))
        #elif isinstance(chan,int):
            #chan=[chan]

        #if num_sweep>(int(self.header['NR'])-1):
            #raise ValueError('Error : there are only '+self.header['NR']+' sweeps recorded in the file.')
        #elif any([x>int(self.header['NC']) for x in chan]):
            #raise ValueError('Error : there are only '+self.header['NC']+' channels recorded in the file')

        #datafile=open(self.fileName,'rb')
        
        ##seek the beginning of the analysis block
        #datafile.seek(1024+num_sweep*512*(int(self.header['NBD'])+int(self.header['NBA'])))
        #_sweepInfo= [datafile.read(8)] #ACCEPTED or REJECTED
        #_sweepInfo+=[datafile.read(4)] #TEST, LEAK, or other name
        #_sweepInfo+=list(np.fromfile(datafile,dtype=np.float32,count=(3+int(self.header['NC'])),sep=""))
        ##sweep info is [status, type, group number, time of beginning, sampling interval, max positive A/D voltage for each channel
        
        #if data:
             ##seek the beginning of the sweep
            #datafile.seek(1024+int(self.header['NBA'])*512+num_sweep*512*(int(self.header['NBD'])+int(self.header['NBA'])))
            #trace=np.fromfile(datafile,dtype=np.int16,count=(int(self.header['NBD'])*256),sep="")
            ##reshape the array to have one line per channel, flipub to match the number of the line and the number of channel
            ##TO DO replace flipud with info from header (Y0 or YR)
            #trace = trace.astype('float64')
            #trace=np.reshape(trace,(int(self.header['NC']),-1),order='F')
            
            ##Make trace, an array of floating number which is the output of the function
            ##print trace
            ##print [_sweepInfo[5+i] for i in range(int(self.header['NC']))]
            ##print float(self.header['ADCMAX'])
            ##print [ float(self.header['YG'+str(i)]) for i in range(int(self.header['NC']))]
            #scale = [_sweepInfo[5+i]/float(self.header['YG'+str(i)])/float(self.header['ADCMAX'])  \
            #for i in range(int(self.header['NC']))]
            #if not self.policy.has_key('memmap'):
                #self._remember(num_sweep, trace, scale = scale, dt = _sweepInfo[4])
            #for i in range(int(self.header['NC'])):
                    #trace[i,:] *= scale[i]
                    ##convert data according to the gain from the analysis block
            ##for i in chan:
                ##if _sweepInfo[5+i]/float(self.header['YG'+str(i)])!=self.channelInfo[i].maxval:
                    ##print 'Error !! The maximum positive limit for channel '+str(i)+' has changed. Check that the gain is correct.'
                    ###printed when the gain is changed during the recording. Shouldn't be a problem execpt if it changes during a sweep
            
            #time=np.arange(trace.shape[1])*_sweepInfo[4]
            #trace=np.vstack((time,trace))
                    ##Add a first line for the time
            #chan=[i+1 for i in chan]
            #trace=np.asarray(trace[[0]+chan,:])
            
        #if info:
            #channelInfo=[]
            #for i in range (int(self.header['NC'])):
                #name = 'Ch_'+str(i)
                #title = self.header['YN'+str(i)]
                #units = self.header['YU'+str(i)]
                #isinput = False
                #maxval = _sweepInfo[5+i]
                #gain = self.header['YG'+str(i)]
                #channelInfo.append(ChannelType(name, title, units, isinput,   \
                #maxval, gain))
            
            ##keep only the channel in 'chan'
            #outinfo = {'t0':_sweepInfo[3] , 'sampling' : float(_sweepInfo[4]),\
            #'numChans' : int(self.header['NC']), 'channelInfo' : channelInfo}
    
        #if info&data:
            #return [trace, outinfo]
        #elif info:
            #return outinfo
        #elif data:
            #return trace

