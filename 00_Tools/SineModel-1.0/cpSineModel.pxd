#this is a cython wrapper on C functions to call them in python

cdef extern from "cSineModel.h":
    ctypedef struct ConfParm:
        double freqDevSlope
        int freqDevOffset
        int MagCond
        int numFrames
        int numBins
        double mindB
        double maxdB
        double binFreq

    ctypedef struct Partial:
        int period[2]
        double *mag
        double *freq
        int *magIdx
        int *freqIdx
        int size
        int *type
        int TracksStatus
        
    void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec)
    Partial *PT_Algo_FM_C(double *mXdB, double *ploc, double *vploc, ConfParm Parm, int *NumOfTracks)