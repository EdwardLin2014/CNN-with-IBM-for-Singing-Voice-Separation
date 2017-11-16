#########################################################################
## Step 0 - Import Library
import os, sys
import time
import numpy as np
import scipy.signal as sg
import h5py
import resampy as rs

Tool_UtilFunc_DirStr = '../00_Tools/UtilFunc-1.0/'
Tool_SineModel_DirStr = '../00_Tools/SineModel-1.0/'
WavDirStr = './Wavfile/Sources/'
H5DirStr = '../00_HDF5/'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_SineModel_DirStr))
import Database as DB
import Audio as AD
from structDT import Param
from structDT import Signal
import SineModel as SM

def ismember(A, B):
    return np.sum(A == B)

#########################################################################
## Step 0 - Parmaters Setting
## Obtain Audio File Name
TestNames, DevNames = DB.DSD100RawNames(WavDirStr)
AllNames = np.concatenate((TestNames, DevNames))
# STFT
Parm = Param()
Parm.M = 1024                                      # Window Size, 23.22ms
Parm.window = sg.get_window('hann', Parm.M)        # Window in Vector Form
Parm.N = 4096                                      # Analysis DFT Size, 92.88ms
Parm.H = 256                                       # Hop Size, 5.80ms
Parm.fs = 22050                                    # Sampling Rate, 22.05K Hz
Parm.numFrames = 2584                              # Number of frames in spectrogram
Parm.numBins = 2049                                # Number of frequency bin in spectrogram
#FCN
numFTBins = 18441                                  # 2049(0~11.025kHz) * 9 (~104.49 msec)
numIFrames = 9                                     # Each input for CNN is ~104.49 msec
hNumIFrames = int(np.floor(numIFrames/2))          # 4
hopNumFrames = 8                                   # ~92.88 ms
numInputPerSong = int(Parm.numFrames/hopNumFrames) # 323 CNN input instances per one song
Mix_mXL_Data = np.zeros((numFTBins,numInputPerSong))
Voice_mXL_Data = np.zeros((numFTBins,numInputPerSong))
Mix_mXR_Data = np.zeros((numFTBins,numInputPerSong))
Voice_mXR_Data = np.zeros((numFTBins,numInputPerSong))
IBML_label = np.zeros((numFTBins,numInputPerSong))
IBMR_label = np.zeros((numFTBins,numInputPerSong))
Pad0 = np.float32(np.zeros((Parm.numBins,hNumIFrames)))
PadMin = np.float32(np.zeros((Parm.numBins,hNumIFrames)))

Mix_Train_Data = np.zeros((numFTBins,numInputPerSong))
Voice_Train_Data = np.zeros((numFTBins,numInputPerSong))
Mix_Valid_Data = np.zeros((numFTBins,numInputPerSong))
Voice_Valid_Data = np.zeros((numFTBins,numInputPerSong))
IBM_Train_label = np.zeros((numFTBins,numInputPerSong))
IBM_Valid_label = np.zeros((numFTBins,numInputPerSong))
OddIdx = np.arange(0,numInputPerSong,2)
EvenIdx = np.arange(1,numInputPerSong,2)

numMusics = 100
fs = 44100
hfs = fs/2
win = fs*30
ove = fs*15
# Ignore the following clips as its Vocal SDR is NaN (Not a Number)
isNanIdx = [-1]*100
isNanIdx[4] = 0
isNanIdx[6] = 8
isNanIdx[7] = [0,1,2]
isNanIdx[10] = 0
isNanIdx[12] = [16,17]
isNanIdx[15] = [0,12,13,14,15]
isNanIdx[16] = [0,1,16,17,18,19]
isNanIdx[18] = [0,1]
isNanIdx[19] = 10
isNanIdx[21] = [0,9,10]
isNanIdx[25] = [0,9,15]
isNanIdx[28] = [10,17,18,19,20]
isNanIdx[29] = [0,14,15]
isNanIdx[30] = [9,10,11,12]
isNanIdx[32] = [0,11,16,17,18,19]
isNanIdx[35] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
isNanIdx[36] = [0,5]
isNanIdx[37] = [0,9,10,11,12]
isNanIdx[38] = 0
isNanIdx[39] = 0
isNanIdx[40] = [0,8,9]
isNanIdx[43] = [0,1,2]
isNanIdx[44] = [16,17,18,19,20,21,22,23,24,25,26]
isNanIdx[48] = 0
isNanIdx[55] = 12
isNanIdx[56] = [13,14]
isNanIdx[57] = 14
isNanIdx[58] = [0,1,10,11]
isNanIdx[59] = [0,1]
isNanIdx[60] = [0,1]
isNanIdx[61] = 0
isNanIdx[62] = [0,10]
isNanIdx[63] = 0
isNanIdx[64] = [0,1,2]
isNanIdx[65] = [0,1,2,14,15,16,17,18,19]
isNanIdx[66] = [0,17,22,23,24,25,26]
isNanIdx[67] = [0,1,2,3,4,5]
isNanIdx[68] = 0
isNanIdx[69] = 2
isNanIdx[70] = [14,15]
isNanIdx[71] = [0,1,2]
isNanIdx[72] = 9
isNanIdx[73] = 14
isNanIdx[76] = 0
isNanIdx[78] = [4,5,10,11,15,16]
isNanIdx[80] = [5,6,13,14,15]
isNanIdx[83] = [10,11]
isNanIdx[85] = [0,1]
isNanIdx[86] = [0,1,2,10,11]
isNanIdx[87] = [0,1,2,3,4,9]
isNanIdx[88] = [0,6,7]
isNanIdx[89] = 0
isNanIdx[90] = [0,18]
isNanIdx[91] = 19
isNanIdx[92] = 0
isNanIdx[93] = 0
isNanIdx[94] = 27
isNanIdx[97] = 10
isNanIdx[98] = [4,9,10]
isNanIdx[99] = 12
nwin = np.zeros((numMusics,1))

TrainMusic = np.arange(50,100)
ValidMusic = np.arange(50,100)
TestMusic = np.arange(50)

# h5files - Total 677 Musics
trainFrames = 347*323       # 112,081    Left Odd, Right Even
validFrames = 347*323       # 112,081    Right Odd, Left Even
testFrames = 298*323        # 96,254     Left Channel, Music 1-50, except 36,37,43,44
H5_Write = H5DirStr + 'DSD100_IBM.h5'
h5fCNNOutput = h5py.File(H5_Write, 'w')
train = h5fCNNOutput.create_dataset("train",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
trainLabel = h5fCNNOutput.create_dataset("trainLabel",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
trainAE = h5fCNNOutput.create_dataset("trainAE",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
valid = h5fCNNOutput.create_dataset("valid",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
validLabel = h5fCNNOutput.create_dataset("validLabel",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
validAE = h5fCNNOutput.create_dataset("validAE",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
test = h5fCNNOutput.create_dataset("test",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
testLabel = h5fCNNOutput.create_dataset("testLabel",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
testAE = h5fCNNOutput.create_dataset("testAE",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')

Mix = Signal()
Voice = Signal()
Song = Signal()
TrainIdx = 0
ValidIdx = 0
TestIdx = 0
for t in np.arange(numMusics):
    
    # Ignore the following Songs as they are exempted in the standard evaluation
    if t == 35 or t == 36 or t == 42 or t == 43:
        continue
    
    #########################################################################
    ## Step 1 - Import Audio and Create Spectrogram
    tic = time.time()
    if t<50:
        SetStr = 'Test/'
    else:
        SetStr = 'Dev/'
    BassPath = WavDirStr + SetStr + AllNames[t] + '/bass.wav'
    DrumsPath = WavDirStr + SetStr + AllNames[t] + '/drums.wav'
    OtherPath = WavDirStr + SetStr + AllNames[t] + '/other.wav'
    VoicePath = WavDirStr + SetStr + AllNames[t] + '/vocals.wav'
    # Import Audio
    Bassx, _ = AD.audioread(BassPath)
    BassxL = Bassx[:,0]    
    Drumx, _ = AD.audioread(DrumsPath)
    DrumxL = Drumx[:,0]    
    Otherx, _ = AD.audioread(OtherPath)
    OtherxL = Otherx[:,0]    
    Voice.x, _ = AD.audioread(VoicePath)
    Voice.xL = Voice.x[:,0]
    Song.xL = BassxL + DrumxL + OtherxL
    # In Dev Set, also use right channel as Validation
    if t>49:
        BassxR = Bassx[:,1]
        DrumxR = Drumx[:,1]
        OtherxR = Otherx[:,1]
        Voice.xR = Voice.x[:,1]
        Song.xR = BassxR + DrumxR + OtherxR
    toc = time.time() - tic
    print('Import audio - %d:%s - needs %.2f sec' % (t, AllNames[t], toc) )
    
    nsampl = len(Song.xL)
    nwin[t] = np.floor((nsampl-win+1+ove)/ove)
    for k in np.arange(nwin[t]):
        if ismember(k, isNanIdx[t]):
            continue
        if k%2 == 0:    # Want even clips
            continue
        startIdx = int(k*ove)
        endIdx = int(startIdx+win)
        #########################################################################    
        ## Step 2 - resample + STFT
        tic = time.time()
        VoicexL = rs.resample(Voice.xL[startIdx:endIdx], fs, hfs)
        SongxL = rs.resample(Song.xL[startIdx:endIdx], fs, hfs)
        MixxL = VoicexL + SongxL
        # STFT
        _, Voice.mXL, _, _, _, _ = SM.stft(VoicexL, Parm)
        _, Song.mXL, _, _, _, _ = SM.stft(SongxL, Parm)
        _, Mix.mXL, _, _, _, _ = SM.stft(MixxL, Parm)
        Voice.mXL = np.float32(Voice.mXL)
        Song.mXL = np.float32(Song.mXL)
        Mix.mXL = np.float32(Mix.mXL)
        # In Dev Set, also use right channel as Validation
        if t>49:
            VoicexR = rs.resample(Voice.xR[startIdx:endIdx], fs, hfs)
            SongxR = rs.resample(Song.xR[startIdx:endIdx], fs, hfs)
            MixxR = VoicexR + SongxR
            # STFT
            _, Voice.mXR, _, _, _, _ = SM.stft(VoicexR, Parm)
            _, Song.mXR, _, _, _, _ = SM.stft(SongxR, Parm)
            _, Mix.mXR, _, _, _, _ = SM.stft(MixxR, Parm)
            Voice.mXR = np.float32(Voice.mXR)
            Song.mXR = np.float32(Song.mXR)
            Mix.mXR = np.float32(Mix.mXR)
        toc = time.time() - tic
        print('STFT - %d:%d - needs %.2f sec' % (t, k, toc) )
        
        #########################################################################    
        ## Step 3 - Prepare Data and Label
        tic = time.time()
        PadMin[:,:] = min(Mix.mXL.min(0))
        MixmXL = np.concatenate((PadMin,Mix.mXL,PadMin),axis=1)
        IBML = Voice.mXL>Song.mXL
        VoicemXL = np.concatenate((PadMin,Mix.mXL*IBML,PadMin),axis=1)
        IBML = np.concatenate((Pad0,IBML,Pad0),axis=1)
        IBML[IBML==0] = 0.02
        IBML[IBML==1] = 0.98
        # In Dev Set, also use right channel as Validation
        if t>49:
            PadMin[:,:] = min(Mix.mXR.min(0))
            MixmXR = np.concatenate((PadMin,Mix.mXR,PadMin),axis=1)
            IBMR = Voice.mXR>Song.mXR
            VoicemXR = np.concatenate((PadMin,Mix.mXR*IBMR,PadMin),axis=1)
            IBMR = np.concatenate((Pad0,IBMR,Pad0),axis=1)
            IBMR[IBMR==0] = 0.02
            IBMR[IBMR==1] = 0.98
        toc = time.time() - tic
        print('Prepare Data and Label - %d:%d - needs %.2f sec' % (t, k, toc) )

        #########################################################################
        ## Step 4 - Prepare CNN Input
        tic = time.time()
        for i in range(numInputPerSong):
            startIdx = i*hopNumFrames
            endIdx = startIdx + numIFrames
            Mix_mXL_Data[:,i:i+1] = np.reshape(MixmXL[:,startIdx:endIdx],(numFTBins,1),order="F")
            Voice_mXL_Data[:,i:i+1] = np.reshape(VoicemXL[:,startIdx:endIdx],(numFTBins,1),order="F")
            IBML_label[:,i:i+1] = np.reshape(IBML[:,startIdx:endIdx],(numFTBins,1),order="F")
            # In Dev Set, also use right channel as Validation
            if t>49:
                Mix_mXR_Data[:,i:i+1] = np.reshape(MixmXR[:,startIdx:endIdx],(numFTBins,1),order="F")
                Voice_mXR_Data[:,i:i+1] = np.reshape(VoicemXR[:,startIdx:endIdx],(numFTBins,1),order="F")
                IBMR_label[:,i:i+1] = np.reshape(IBMR[:,startIdx:endIdx],(numFTBins,1),order="F")
        toc = time.time() - tic
        print('Prepare CNN Input - %d:%d - needs %.2f sec' % (t, k, toc) )

        #########################################################################
        ## Step 5 - Save in h5Files
        tic = time.time()
        colIdxs = np.random.permutation(numInputPerSong)
        # Dev Set: Training and Validation
        if t>49:
            Mix_Train_Data[:,OddIdx] = Mix_mXL_Data[:,OddIdx]
            Mix_Train_Data[:,EvenIdx] = Mix_mXR_Data[:,EvenIdx]
            Mix_Train_Data = Mix_Train_Data[:,colIdxs]
            IBM_Train_label[:,OddIdx] = IBML_label[:,OddIdx]
            IBM_Train_label[:,EvenIdx] = IBMR_label[:,EvenIdx]
            IBM_Train_label = IBM_Train_label[:,colIdxs]
            Voice_Train_Data[:,OddIdx] = Voice_mXL_Data[:,OddIdx]
            Voice_Train_Data[:,EvenIdx] = Voice_mXR_Data[:,EvenIdx]
            Voice_Train_Data = Voice_Train_Data[:,colIdxs]
            
            Mix_Valid_Data[:,EvenIdx] = Mix_mXL_Data[:,EvenIdx]
            Mix_Valid_Data[:,OddIdx] = Mix_mXR_Data[:,OddIdx]
            Mix_Valid_Data = Mix_Valid_Data[:,colIdxs]
            IBM_Valid_label[:,EvenIdx] = IBML_label[:,EvenIdx]
            IBM_Valid_label[:,OddIdx] = IBMR_label[:,OddIdx]
            IBM_Valid_label = IBM_Valid_label[:,colIdxs]
            Voice_Valid_Data[:,EvenIdx] = Voice_mXL_Data[:,EvenIdx]
            Voice_Valid_Data[:,OddIdx] = Voice_mXR_Data[:,OddIdx]
            Voice_Valid_Data = Voice_Valid_Data[:,colIdxs]            
        
            start = TrainIdx*numInputPerSong
            end = (TrainIdx+1)*numInputPerSong
            train[start:end,:] = np.transpose(Mix_Train_Data)
            trainLabel[start:end,:] = np.transpose(IBM_Train_label)
            trainAE[start:end,:] = np.transpose(Voice_Train_Data)
            TrainIdx += 1
            
            start = ValidIdx*numInputPerSong
            end = (ValidIdx+1)*numInputPerSong
            valid[start:end,:] = np.transpose(Mix_Valid_Data)
            validLabel[start:end,:] = np.transpose(IBM_Valid_label)
            validAE[start:end,:] = np.transpose(Voice_Valid_Data)
            ValidIdx += 1
        else:
            # Test Set: Testing
            Mix_mX_Data = Mix_mXL_Data[:,colIdxs]
            IBM_label = IBML_label[:,colIdxs]
            Voice_mX_Data = Voice_mXL_Data[:,colIdxs]
            start = TestIdx*numInputPerSong
            end = (TestIdx+1)*numInputPerSong
            test[start:end,:] = np.transpose(Mix_mX_Data)
            testLabel[start:end,:] = np.transpose(IBM_label)
            testAE[start:end,:] = np.transpose(Voice_mX_Data)
            TestIdx += 1
            
        print('Save in h5Files - %d:%d - needs %.2f sec' % (t, k, toc))
        
h5fCNNOutput.close()