#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#ifndef CSINEMODEL_H    
#define CSINEMODEL_H

#define BH_SIZE 1001
#define BH_SIZE_BY2 501

typedef struct _ConfParm
{
    double freqDevSlope;
    int freqDevOffset;
    int MagCond;
    int numFrames;
    int numBins;
    double mindB;
    double maxdB;
    double binFreq;
} ConfParm;

typedef struct _Partial
{
    int period[2];
    double *mag;
    double *freq;
    int *magIdx;
    int *freqIdx;
    int size;
    int *type;
    // 0: Not Selected, Active
    // 1: Selected
    // -1: Deactive
    int TracksStatus;
} Partial;

typedef struct _Peak
{
    double mag;
    double freq;
    double magIdx;
    double freqIdx;
    double type;
    int NotSelectedPeak;
    int SelectedTracks;
    int PartialsIdx;
} Peak;

void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec);

int dBToMagLvl(double dB, double mindB, double maxdB);
Peak *initPeaks(int numPeaks, double *mXdB, int *idxs, int numBins, int curFrame, double binFreq, double *vploc,double mindB, double maxdB);
int cmpfunc (const void *val1, const void *val2);
int *findPeak(double *array, int numBins, int *numPeaks );
int isAnyActiveTracks(Partial *Partials, int NumOfTracks);
Peak *getPrevPeaks(Partial *Partials, int NumOfTracks, int *numPrevPeaks);
Partial *PT_Algo_FM_C(double *mXdB, double *ploc, double *vploc, ConfParm Parm, int *NumOfTracks);

#endif  //CSINEMODEL_H