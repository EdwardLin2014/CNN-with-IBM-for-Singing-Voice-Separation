#this is a cython wrapper on C functions to call them in python

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from cpSineModel cimport *

class PElm:    
    def __init__(self):
        self.period = []
        self.mag = []
        self.freq = []
        self.magIdx = []
        self.freqIdx = []
        self.size = 1
        self.type = []

def genSpecSines(iploc, ipmag, ipphase, N):
    "comments"
    
    cdef np.ndarray[np.float_t, ndim=1] iploc_arr
    cdef np.ndarray[np.float_t, ndim=1] ipmag_arr
    cdef np.ndarray[np.float_t, ndim=1] ipphase_arr
    cdef np.ndarray[np.float_t, ndim=1] real_arr
    cdef np.ndarray[np.float_t, ndim=1] imag_arr
        
    iploc_arr = np.ascontiguousarray(iploc, dtype=np.float)
    ipmag_arr = np.ascontiguousarray(ipmag, dtype=np.float)
    ipphase_arr = np.ascontiguousarray(ipphase, dtype=np.float)
    
    real_arr = np.zeros((N,), dtype=np.float)
    imag_arr = np.zeros((N,), dtype=np.float)
        
    genspecsines_C(<double *>iploc_arr.data, <double *>ipmag_arr.data, <double *>ipphase_arr.data, iploc_arr.shape[0],  <double *>real_arr.data,  <double *>imag_arr.data, N)
    
    out = real_arr.astype(complex)
    out.imag = imag_arr
    return out

def PT_Algo_FM(inmXdB, inploc, invploc, inParm):
    "comments"

    ## Input Variables
    cdef ConfParm Parm
    Parm.freqDevSlope = inParm.freqDevSlope
    Parm.freqDevOffset = inParm.freqDevOffset
    Parm.MagCond = inParm.MagCond
    Parm.numFrames = inParm.numFrames
    Parm.numBins = inParm.numBins
    Parm.mindB = inParm.mindB
    Parm.maxdB = inParm.maxdB
    minPartialLength = inParm.minPartialLength
    Parm.binFreq = inParm.fs/inParm.N
    
    ## Output Variables
    cdef int NumOfTracks = 0
    cdef np.ndarray[np.float_t, ndim=2] mXdB
    cdef np.ndarray[np.float_t, ndim=2] ploc
    cdef np.ndarray[np.float_t, ndim=2] vploc
    mXdB = np.ascontiguousarray(inmXdB, dtype=np.float)
    ploc = np.ascontiguousarray(inploc, dtype=np.float)
    vploc = np.ascontiguousarray(invploc, dtype=np.float)
    
    cPartials = PT_Algo_FM_C(<double *>mXdB.data, <double *>ploc.data, <double *>vploc.data, Parm, &NumOfTracks)
    
    ## create a NumOfPartials x 1 struct matrix for output
    NumOfPartials = 0;
    for i in np.arange(NumOfTracks):
        if(cPartials[i].size >= minPartialLength):
            NumOfPartials += 1
    Partials = [[]] * NumOfPartials;
    ## Obtain Partials and pass to python code
    j = 0
    for i in np.arange(NumOfTracks):
        if(cPartials[i].size >= minPartialLength):
            elm = PElm()
            ## period
            elm.period.append(cPartials[i].period[0])
            elm.period.append(cPartials[i].period[1])
            ## mag
            for k in np.arange(cPartials[i].size):
                elm.mag.append(cPartials[i].mag[k])
            ## freq
            for k in np.arange(cPartials[i].size):
                elm.freq.append(cPartials[i].freq[k])
            ## magIdx
            for k in np.arange(cPartials[i].size):
                elm.magIdx.append(cPartials[i].magIdx[k])
            ## freqIdx
            for k in np.arange(cPartials[i].size):
                elm.freqIdx.append(cPartials[i].freqIdx[k])
            ## size
            elm.size = cPartials[i].size
            ## type
            for k in np.arange(cPartials[i].size):
                elm.type.append(cPartials[i].type[k])
            ## SetCell
            Partials[j] = elm
            j += 1

        free(cPartials[i].mag)
        free(cPartials[i].magIdx)
        free(cPartials[i].freq)
        free(cPartials[i].freqIdx)
        free(cPartials[i].type)

    if(cPartials):
        free(cPartials)    

    return Partials
