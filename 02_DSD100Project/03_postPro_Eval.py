import dsdtools

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
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_SineModel_DirStr))
ModelDirStr = './model_20171006_0945'
from structDT import Param
from structDT import Signal
import SineModel as SM

#########################################################################
## Step 0 - Parmaters Setting
Threshold = 0.15                                    # Polish Vocal Mask further
# STFT
Parm = Param()
Parm.M = 1024                                       # Window Size, 46.44ms
Parm.window = sg.get_window('hann', Parm.M)         # Window in Vector Form
Parm.N = 4096                                       # Analysis DFT Size, 185.76ms
Parm.H = 256                                        # Hop Size, 11.61ms
Parm.fs = 22050                                     # Sampling Rate, 22.05K Hz
#CNN
numFTBins = 18441                                   # 2049(0~11.025kHz) * 9 (~104.49 msec)
numIFrames = 9                                      # Each input for CNN is ~104.49 msec
hNumIFrames = int(np.floor(numIFrames/2))           # 4
Mix = Signal()
Voice = Signal()
Song = Signal()

#########################################################################
## Step 0 - Restore the model
tic = time.time()
sess = tf.Session() 
saver = tf.train.import_meta_graph(ModelDirStr+'/model-279.meta')
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

#########################################################################
## Step 1 - Actual Separation, used by dsdtools package
def my_function(track):
    '''My fancy BSS algorithm'''

    # get the audio mixture as numpy array shape=(nun_sampl, 2)
    x = track.audio
    # get the sample rate
    fs = track.rate
    #################################################################
    # Left Channel
    Mix.x = x[:,0]
    numSamples = len(Mix.x)
    Mix.x = rs.resample(Mix.x, fs, fs/2)
    _,Mix.mX,Mix.pX,Parm.remain,Parm.numFrames,Parm.numBins = SM.stft(Mix.x, Parm)
    Mix.mX = np.float32(Mix.mX)
    Mix.pX = np.float32(Mix.pX)
    PadMin = np.zeros((Parm.numBins,hNumIFrames))
    
    PadMin[:,:] = min(Mix.mX.min(0))
    CNNmX = np.concatenate((PadMin, Mix.mX, PadMin), axis=1)
    CNNInput = np.zeros((Parm.numFrames,numFTBins))
    for i in range(Parm.numFrames):
        startIdx = i
        endIdx = startIdx + numIFrames
        CNNInput[i:i+1,:] = np.transpose(np.reshape(CNNmX[:,startIdx:endIdx],(numFTBins,1),order="F"))
    
    VoiceSigMap = np.zeros((Parm.numBins,Parm.numFrames))
    VoiceMask = np.zeros((Parm.numBins,Parm.numFrames))
    for i in range(Parm.numFrames):
        tmpSig = sess.run(ySig, feed_dict={xIn:CNNInput[i:i+1,:], keep_prob:1.0})
        tmpSigMap = np.transpose(np.reshape(tmpSig,(numIFrames,Parm.numBins)))
        VoiceSigMap[:,i] = tmpSigMap[:,hNumIFrames]
    SongSigMap = 1 - VoiceSigMap
    VoiceMask = VoiceSigMap
    SongMask = SongSigMap
    VoiceMask[VoiceMask<Threshold] = 0
    SongMask[SongMask<Threshold] = 0
    
    mV = np.multiply(VoiceMask, Mix.mX)
    Voice.y = SM.istft(mV, Mix.pX, Parm )
    Voice.y = rs.resample(Voice.y,fs/2,fs)
    Voice.y = Voice.y[0:numSamples]    
    
    mS = np.multiply(SongMask, Mix.mX)
    Song.y = SM.istft(mS, Mix.pX, Parm )
    Song.y = rs.resample(Song.y,fs/2,fs)
    Song.y = Song.y[0:numSamples]
    
    voc_array = np.zeros((numSamples,2))
    voc_array[:,0] = Voice.y
    acc_array = np.zeros((numSamples,2))
    acc_array[:,0] = Song.y
    
    #################################################################
    # Right Channel
    Mix.x = x[:,1]
    numSamples = len(Mix.x)
    Mix.x = rs.resample(Mix.x, fs, fs/2)
    _,Mix.mX,Mix.pX,Parm.remain,Parm.numFrames,Parm.numBins = SM.stft(Mix.x, Parm)
    Mix.mX = np.float32(Mix.mX)
    Mix.pX = np.float32(Mix.pX)
    PadMin = np.zeros((Parm.numBins,hNumIFrames))
    
    PadMin[:,:] = min(Mix.mX.min(0))
    CNNmX = np.concatenate((PadMin, Mix.mX, PadMin), axis=1)
    CNNInput = np.zeros((Parm.numFrames,numFTBins))
    for i in range(Parm.numFrames):
        startIdx = i
        endIdx = startIdx + numIFrames
        CNNInput[i:i+1,:] = np.transpose(np.reshape(CNNmX[:,startIdx:endIdx],(numFTBins,1),order="F"))
    
    VoiceSigMap = np.zeros((Parm.numBins,Parm.numFrames))
    VoiceMask = np.zeros((Parm.numBins,Parm.numFrames))
    for i in range(Parm.numFrames):
        tmpSig = sess.run(ySig, feed_dict={xIn:CNNInput[i:i+1,:], keep_prob:1.0})
        tmpSigMap = np.transpose(np.reshape(tmpSig,(numIFrames,Parm.numBins)))
        VoiceSigMap[:,i] = tmpSigMap[:,hNumIFrames]
    SongSigMap = 1 - VoiceSigMap
    VoiceMask = VoiceSigMap
    SongMask = SongSigMap
    VoiceMask[VoiceMask<Threshold] = 0
    SongMask[SongMask<Threshold] = 0
    
    mV = np.multiply(VoiceMask, Mix.mX)
    Voice.y = SM.istft(mV, Mix.pX, Parm )
    Voice.y = rs.resample(Voice.y,fs/2,fs)
    Voice.y = Voice.y[0:numSamples]    
    
    mS = np.multiply(SongMask, Mix.mX)
    Song.y = SM.istft(mS, Mix.pX, Parm )
    Song.y = rs.resample(Song.y,fs/2,fs)
    Song.y = Song.y[0:numSamples]

    voc_array[:,1] = Voice.y
    acc_array[:,1] = Song.y

    # return any number of targets
    estimates = {
        'vocals': voc_array,
        'accompaniment': acc_array,
    }
    return estimates

# initiate dsdtools
dsd = dsdtools.DB(root_dir="./Wavfile")

# verify if my_function works correctly
if dsd.test(my_function):
    print("my_function is valid")

dsd.run(
    my_function,
    estimates_dir='./Audio/CNN_015',
)
