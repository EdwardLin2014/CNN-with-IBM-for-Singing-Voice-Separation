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
WavDirStr = '../Wavfile/'                      # Amend this line!
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
WavFileNames = DB.iKalaWavFileNames(WavDirStr)
numMusics = len(WavFileNames)
numSamples = 1323008
# STFT
Parm = Param()
Parm.M = 1024                                      # Window Size, 46.44ms
Parm.window = sg.get_window('hann', Parm.M)        # Window in Vector Form
Parm.N = 4096                                      # Analysis DFT Size, 185.76ms
Parm.H = 256                                       # Hop Size, 11.61ms
Parm.fs = 22050                                    # Sampling Rate, 22.05K Hz
Parm.numFrames = 2584                              # Number of frames in spectrogram
Parm.numBins = 2049                                # Number of frequency bin in spectrogram
fs = 44100                                         # sampling rate of Wave file
hfs = fs/2
# Dataset - See paper for details
trainMusic = [3,4,5,6,7,8,9,10,11,13,14,16,17,20,21,23,27,29,30,33,34,35,36,37,38,39,41,44,46,50,52,53,54,55,56,57,59,60,61,63,64,65,66,67,70,71,73,77,78,82,84,85,86,88,91,92,95,96,97,98,102,103,108,109,114,115,116,117,119,123,124,126,127,128,130,133,141,142,144,145,146,147,149,150,151,153,154,155,156,158,159,160,162,163,164,165,169,170,172,173,174,176,180,184,185,186,187,188,189,190,191,193,194,196,197,199,200,201,203,207,208,209,210,211,215,216,217,219,221,224,225,226,227,230,231,232,233,234,235,236,237,238,239,240,241,242,246,248,249,250,251,252]
valMusic = [1,22,28,32,42,48,49,58,62,72,74,80,83,89,90,93,101,106,120,121,122,125,129,131,136,140,143,148,157,161,168,178,181,182,183,192,195,198,205,206,212,213,214,220,222,229,243,244,245,247]
testMusic = [2,12,15,18,19,24,25,26,31,40,43,45,47,51,68,69,75,76,79,81,87,94,99,100,104,105,107,110,111,112,113,118,132,134,135,137,138,139,152,166,167,171,175,177,179,202,204,218,223,228]
#CNN
numFTBins = 18441                                  # 2049(0~11.025kHz) * 9 (~104.49 msec)
numIFrames = 9                                     # Each input for CNN is ~104.49 msec
hNumIFrames = int(np.floor(numIFrames/2))          # 4
hopNumFrames = 8                                   # ~92.88 ms
numInputPerSong = int(Parm.numFrames/hopNumFrames) # 323 CNN input instances per one song
NetData = np.zeros((numFTBins,numInputPerSong))
AEData = np.zeros((numFTBins,numInputPerSong))
label = np.zeros((numFTBins,numInputPerSong))
Pad0 = np.float32(np.zeros((Parm.numBins,hNumIFrames)))
PadMin = np.float32(np.zeros((Parm.numBins,hNumIFrames)))
# h5files
TrainIdx = 0
ValidIdx = 0
TestIdx = 0
trainFrames = 152*323      # 49,096
validFrames = 50*323       # 16,150
testFrames = 50*323        # 16,150
H5_Write = H5DirStr + 'iKala_IBM.h5'
h5fContent = h5py.File(H5_Write, 'w')
train = h5fContent.create_dataset("train",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
trainLabel = h5fContent.create_dataset("trainLabel",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
trainAE = h5fContent.create_dataset("trainAE",(trainFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
valid = h5fContent.create_dataset("valid",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
validLabel = h5fContent.create_dataset("validLabel",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
validAE = h5fContent.create_dataset("validAE",(validFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
test = h5fContent.create_dataset("test",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
testLabel = h5fContent.create_dataset("testLabel",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')
testAE = h5fContent.create_dataset("testAE",(testFrames,numFTBins),chunks=(1,numFTBins),dtype='f')

for t in np.arange(numMusics):
    #########################################################################
    ## Step 1 - Import Audio and Create Power Spectrogram
    tic = time.time()
    x, fs = AD.audioread(WavFileNames[t])
    Mix = Signal()
    Voice = Signal()
    Song = Signal()
    Voice.x = rs.resample(x[:,1], fs, hfs)
    Song.x = rs.resample(x[:,0], fs, hfs)
    Mix.x = Voice.x + Song.x
    # Spectrogram Dimension - Parm.numBins:2049 X Parm.numFrames:2584 = 5,294,616
    _, Voice.mX, _, _, _, _ = SM.stft(Voice.x, Parm)
    _, Song.mX, _, _, _, _ = SM.stft(Song.x, Parm)
    _, Mix.mX, _, _, _, _ = SM.stft(Mix.x, Parm)
    Voice.mX = np.float32(Voice.mX)
    Song.mX = np.float32(Song.mX)
    Mix.mX = np.float32(Mix.mX)
    PadMin[:,:] = min(Mix.mX.min(0))
    mX = np.concatenate((PadMin,Mix.mX,PadMin),axis=1)
    IBM = Voice.mX > Song.mX
    VoicemX = np.concatenate((PadMin,Mix.mX*IBM,PadMin),axis=1)
    IBM = np.concatenate((Pad0,IBM,Pad0),axis=1)
    IBM[IBM==0] = 0.02
    IBM[IBM==1] = 0.98
    toc = time.time() - tic
    if t < 137:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) );
    else:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) );
    
    #########################################################################
    ## Step 2 - Prepare CNN Input
    tic = time.time()
    for i in range(numInputPerSong):
        startIdx = i*hopNumFrames
        endIdx = startIdx + numIFrames
        NetData[:,i:i+1] = np.reshape(mX[:,startIdx:endIdx],(numFTBins,1),order="F")
        AEData[:,i:i+1] = np.reshape(VoicemX[:,startIdx:endIdx],(numFTBins,1),order="F")
        label[:,i:i+1] = np.reshape(IBM[:,startIdx:endIdx],(numFTBins,1),order="F")
    toc = time.time() - tic
    print('Prepare CNN Input needs %.2f sec' % toc )

    #########################################################################
    ## Step 3 - Save in h5Files
    # Shuffle the training instances within a song to give regularization
    colIdxs = np.random.permutation(numInputPerSong);
    NetData = NetData[:,colIdxs]
    AEData = AEData[:,colIdxs]
    label = label[:,colIdxs]
    T = t+1
    if ismember(T,trainMusic):
        start = TrainIdx*numInputPerSong
        end = (TrainIdx+1)*numInputPerSong
        train[start:end,:] = np.transpose(NetData)
        trainLabel[start:end,:] = np.transpose(label)
        trainAE[start:end,:] = np.transpose(AEData)
        TrainIdx += 1
    if ismember(T,valMusic):
        start = ValidIdx*numInputPerSong
        end = (ValidIdx+1)*numInputPerSong
        valid[start:end,:] = np.transpose(NetData)
        validLabel[start:end,:] = np.transpose(label)
        validAE[start:end,:] = np.transpose(AEData)
        ValidIdx += 1
    if ismember(T,testMusic):
        start = TestIdx*numInputPerSong
        end = (TestIdx+1)*numInputPerSong
        test[start:end,:] = np.transpose(NetData)
        testLabel[start:end,:] = np.transpose(label)
        testAE[start:end,:] = np.transpose(AEData)
        TestIdx += 1
    if t < 137:
        print('Write hdf5 - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) );
    else:
        print('Write hdf5 - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) );    
    print('--------------------------------------------------------------')
     
h5fContent.close()
    