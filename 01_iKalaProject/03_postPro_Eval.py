#########################################################################
## Step 0 - Import Library
import os, sys
import time
import numpy as np
import scipy.signal as sg
import tensorflow as tf
import resampy as rs

Tool_UtilFunc_DirStr = '../00_Tools/UtilFunc-1.0/'
Tool_SineModel_DirStr = '../00_Tools/SineModel-1.0/'
WavDirStr = './Wavfile/'
OutDirStr = './Audio/Threshold_'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_SineModel_DirStr))
ModelDirStr = './model_20171002_1756'
import Database as DB
import Audio as AD
from structDT import Param
from structDT import Signal
import SineModel as SM

#########################################################################
## Step 0 - Parmaters Setting
## Audio Info
## Obtain Audio File Name
WavFileNames = DB.iKalaWavFileNames(WavDirStr)
numMusics = len(WavFileNames)
numSamples = 1323008
Threshold = 0
# STFT
Parm = Param()
Parm.M = 1024                                       # Window Size, 46.44ms
Parm.window = sg.get_window('hann', Parm.M)         # Window in Vector Form
Parm.N = 4096                                       # Analysis DFT Size, 185.76ms
Parm.H = 256                                        # Hop Size, 11.61ms
Parm.fs = 22050                                     # Sampling Rate, 22.05K Hz
Parm.remain = 255                                   # audio signal between the center of last frame and the end of audio signal; for synthesis
Parm.numFrames = 2584                               # Number of frames in spectrogram
Parm.numBins = 2049                                 # Number of frequency bin in spectrogram
#CNN
numFTBins = 18441                                   # 2049(0~11.025kHz) * 9 (~104.49 msec)
numIFrames = 9                                      # Each input for CNN is ~104.49 msec
hNumIFrames = int(np.floor(numIFrames/2))           # 4
PadMin = np.zeros((Parm.numBins,hNumIFrames))
VoiceSigMap = np.zeros((Parm.numBins,Parm.numFrames))
VoiceMask = np.zeros((Parm.numBins,Parm.numFrames))
CNNInput = np.zeros((Parm.numFrames,numFTBins))
Mix = Signal()
Voice = Signal()
Song = Signal()
    
#########################################################################
## Step 0 - Restore the model
tic = time.time()
sess = tf.Session() 
saver = tf.train.import_meta_graph(ModelDirStr+'/model-70.meta')
saver.restore(sess,tf.train.latest_checkpoint(ModelDirStr))
# Now, let's access and create placeholders variables
graph = tf.get_default_graph()
xIn = graph.get_tensor_by_name("xInput:0")
y_ = graph.get_tensor_by_name("yInput:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
output = graph.get_tensor_by_name("ouputLayer/activation:0")
ySig = tf.nn.sigmoid(output)
toc = time.time() - tic
print('Restore the model needs %.2f sec' % toc)

for t in np.arange(numMusics):
    #########################################################################
    ## Step 1 - STFT
    tic = time.time()
    x, fs = AD.audioread(WavFileNames[t])
    Mix.x = rs.resample(x[:,0]+x[:,1], fs, fs/2)
    # Spectrogram Dimension - Parm.numBins:2049 X Parm.numFrames:2584 = 5,294,616
    _, Mix.mX, Mix.pX, _, _, _ = SM.stft(Mix.x, Parm)
    Mix.mX = np.float32(Mix.mX)
    Mix.pX = np.float32(Mix.pX)
    minmX = min(Mix.mX.min(0))
    toc = time.time() - tic
    if t < 137:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-15:], toc) );
    else:
        print('Import audio - %d:%s - needs %.2f sec' % (t+1, WavFileNames[t][-16:], toc) );
        
    #########################################################################
    ## Step 2 - Prepare CNN Input
    tic = time.time()
    PadMin[:,:] = min(Mix.mX.min(0))
    CNNmX = np.concatenate((PadMin, Mix.mX, PadMin), axis=1)
    for i in range(Parm.numFrames):
        startIdx = i
        endIdx = startIdx + numIFrames
        CNNInput[i:i+1,:] = np.transpose(np.reshape(CNNmX[:,startIdx:endIdx],(numFTBins,1),order="F"))
    toc = time.time() - tic
    print('Prepare CNN Input needs %.2f sec' % toc )
    
    #########################################################################
    ## Step 4 - Calculate Vocal Mask
    tic = time.time()
    for i in range(Parm.numFrames):
        tmpSig = sess.run(ySig, feed_dict={xIn:CNNInput[i:i+1,:], keep_prob:1.0})
        tmpSigMap = np.transpose(np.reshape(tmpSig,(numIFrames,Parm.numBins)))
        VoiceSigMap[:,i] = tmpSigMap[:,hNumIFrames]
    SongSigMap = 1 - VoiceSigMap
    toc = time.time() - tic
    print('Calculate Vocal Map needs %.2f sec' % (toc))

    for j in np.arange(11):
        Threshold = 0.05*j
        VoiceMask = VoiceSigMap
        SongMask = SongSigMap
        VoiceMask[VoiceMask<Threshold] = 0
        SongMask[SongMask<Threshold] = 0
        
        #########################################################################
        ## Step 5 - iSTFT
        tic = time.time()
        mV = np.multiply(VoiceMask, Mix.mX)
        Voice.y = SM.istft(mV, Mix.pX, Parm )
        Voice.y = rs.resample(Voice.y,fs/2,fs)
        Voice.y = Voice.y[0:-2]
        mS = np.multiply(SongMask, Mix.mX)
        Song.y = SM.istft(mS, Mix.pX, Parm )
        Song.y = rs.resample(Song.y,fs/2,fs)
        Song.y = Song.y[0:-2]
        toc = time.time() - tic
        print('%d-%d: iSTFT needs %.2f sec' % (t,j,toc))
        
        #########################################################################
        ## Step 6 - Audio Write
        tic = time.time()
        if t < 137:
            AD.audiowrite( OutDirStr+str(Threshold).replace('.','_')+'/Voice/Voice_'+WavFileNames[t][-15:-4]+'.wav', Voice.y, fs )
            AD.audiowrite( OutDirStr+str(Threshold).replace('.','_')+'/Song/Song_'+WavFileNames[t][-15:-4]+'.wav', Song.y, fs )
        else:
            AD.audiowrite( OutDirStr+str(Threshold).replace('.','_')+'/Voice/Voice_'+WavFileNames[t][-16:-4]+'.wav', Voice.y, fs )
            AD.audiowrite( OutDirStr+str(Threshold).replace('.','_')+'/Song/Song_'+WavFileNames[t][-16:-4]+'.wav', Song.y, fs )
        toc = time.time() - tic
        print('%d-%d: Audio Write needs %.2f sec' % (t,j,toc))
