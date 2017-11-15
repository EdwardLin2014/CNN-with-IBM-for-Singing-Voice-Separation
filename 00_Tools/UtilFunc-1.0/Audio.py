import os, sys
import numpy as np
import copy
from scipy.io.wavfile import read, write
#try:
#	import cAudio
#except ImportError:
#	print("\n")
#	print("-------------------------------------------------------------------------------")
#	print("Warning:")
#	print("Cython modules for some of the core functions were not imported.")
#	print("Please execute the following command under this directory: python compileAudioModule.py build_ext --inplace")
#	print("Exiting the code!!")
#	print("-------------------------------------------------------------------------------")
#	print("\n")
#	sys.exit(0)
    
INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def audioread(filename):
    """
    Read a sound file and convert it to a normalized floating point array
    filename: name of file to read
    returns fs: sampling rate of file, x: floating point array
    """
    # raise error if wrong input file
    if (os.path.isfile(filename) == False):
        raise ValueError("Input file is wrong")
        
    fs, x = read(filename)
    #scale down and convert audio into floating point number in range of -1 to 1
    x = np.float64(x)/norm_fact[x.dtype.name]    
    return x, fs

def audiowrite( filename, y, fs ):
	"""
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""
	x = copy.deepcopy(y)                         # copy array
	x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
	x = np.int16(x)                              # converting to int16 type
	write(filename, fs, x)

def ScaleToRange( NewMin, NewMax, X ):
    OldMin = min(X)
    OldMax = max(X)
    
    ScaledFactor = (NewMax-NewMin)/(OldMax-OldMin)
    ScaledX = ScaledFactor*(X - OldMin) + NewMin
    
    return ScaledX

def scaleAudio( song, MinAmp, MaxAmp ):
    UpperSong = copy.deepcopy(song)
    UpperSong[UpperSong<=0] = 0
    UpperSong[UpperSong>=0] = ScaleToRange( 0, MaxAmp, UpperSong[UpperSong>=0] )
    
    LowerSong = copy.deepcopy(song)
    LowerSong[LowerSong>=0] = 0
    LowerSong[LowerSong<=0] = ScaleToRange( MinAmp, 0, LowerSong[LowerSong<=0] )
    
    Scaledsong = UpperSong + LowerSong
    
    return Scaledsong

#def downsample( x ):
#    y = cAudio.downsample(x)
#    return y
