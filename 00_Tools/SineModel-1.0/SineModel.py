import os, sys
import numpy as np
import math
from scipy.fftpack import fft, ifft, fftshift
Tool_UtilFunc_DirStr = '../UtilFunc-1.0/'
cSineModel_DirStr = '../SineModel-1.0/'
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), Tool_UtilFunc_DirStr))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), cSineModel_DirStr))
import UnitTranslation as UT
import scipy.signal as sg
try:
	import cSineModel
except ImportError:
	print("\n")
	print("-------------------------------------------------------------------------------")
	print("Warning:")
	print("Cython modules for some of the core functions were not imported.")
	print("Please execute the following command under this directory: python compileModule.py build_ext --inplace")
	print("Exiting the code!!")
	print("-------------------------------------------------------------------------------")
	print("\n")
	sys.exit(0)

tol = 1e-14                                                      # threshold used to compute phase

def dftAnal( x, N, window ):
    '''
    %%
    %	Analysis of a signal using the discrete Fourier transform
    %            x: input signal,
    %            N: DFT size (no constraint on power 2!),
    %       window: analysis window (in term of vector)
    %	returns X, mX, pX: complex, magnitude and phase spectrum
    '''
    # window the input sound
    xw = x * window;

    # zero-phase window in fftbuffer
    M = window.size;                            # window Size
    hM1 = int(math.floor((M+1)/2))              # half analysis window size by floor
    hM2 = int(math.floor(M/2))                  # half analysis window size by floor
    fftbuffer = np.zeros(N)                     # initialize buffer for FFT
    if hM2%2 == 1:
        fftbuffer[:hM2] = xw[hM1:]
        fftbuffer[-hM1:] = xw[:hM1]    
    else: # hM1 == hM2
        fftbuffer[:hM1] = xw[hM2:]
        fftbuffer[-hM2:] = xw[:hM2]

    # Compute FFT
    hN = int(math.floor(N/2))+1                # size of positive spectrum, it includes sample 0
    X = fft(fftbuffer) 
    X = X[:hN]
    # for phase calculation set to 0 the small values
    X.real[np.abs(X.real) < tol] = 0.0
    X.imag[np.abs(X.imag) < tol] = 0.0
    # unwrapped phase spectrum of positive frequencies
    pX = np.unwrap(np.angle(X))
    # compute absolute value of positive side
    mX = abs(X)
    
    return X, mX, pX

def stft( x, Parm ):
    '''
    %% Analysis of a sound using the short-time Fourier transform
    %% Input:
    %            x: audio signal as a column vector!
    %         Parm: STFT configuration,
    %  Parm.window: analysis window (in term of vector)
    %       Parm.M: window size,
    %       Parm.N: DFT size (no constraint on power 2!),
    %       Parm.H: hop size,
    %% Ouput:
    %            X: complex spectrogram,
    %           mX: magnitude spectrogram,
    %           pX: phase spectrogram,
    %       remain: audio signal between the center of last frame and the end of
    %                   audio signal; for synthesis
    %    numFrames: number of frames,
    %      numBins: number of bins
    '''
    window = Parm.window
    M = Parm.M
    N = Parm.N
    H = Parm.H
    
    # prepare x
    hM1 = int(math.floor((M+1)/2))             # half analysis window size by floor
    hM2 = int(math.floor(M/2))                 # half analysis window size by floor
    x = np.append(np.zeros(hM2),x)             # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hM1))             # add zeros at the end to analyze last sample
    # prepare window
    window = window / sum(window);             # normalize analysis window

    # prepare stft looping
    pin = list(np.arange(hM2, x.size-hM1, H))
    remain = x.size - hM1 - pin[-1] - 1;

    # prepare output
    numFrames = len(pin);
    hN = int(math.floor(N/2))+1;                # size of positive spectrum, it includes sample 0
    numBins = hN;
    X = np.zeros((hN, numFrames),dtype=complex);
    mX = np.zeros((hN, numFrames));
    pX = np.zeros((hN, numFrames));

    # Note index diff for odd/even size of analysis window
    t = 0;
    if hM2%2 == 1:
        for i in pin:
            X[:,t], mX[:,t], pX[:,t] = dftAnal(x[(i-hM1+1):i+hM2+1], N, window);
            t = t + 1;
    else:
        for i in pin:
            X[:,t], mX[:,t], pX[:,t] = dftAnal(x[(i-hM1):i+hM2], N, window);
            t = t + 1;
    
    return X, mX, pX, remain, numFrames, numBins

def dftSynth( mX, pX, M, N ):
    '''
    %%
    %   Synthesis of a signal using the discrete Fourier transform
    %       mX: magnitude spectrum, 
    %       pX: phase spectrum, 
    %        M: window size,
    %        N: DFT size (no constraint on power 2!)
    %	returns y: output signal
    '''
    hN = mX.size;                                        # size of positive spectrum, it includes sample 0
    hM1 = int(math.floor((M+1)/2))                       # half analysis window size by rounding
    hM2 = int(math.floor(M/2))                           # half analysis window size by floor
    y = np.zeros(M)                                      # initialize output array
    Y = np.zeros(N,dtype=complex)                        # clean output spectrum

    Y[:hN] = mX * np.exp(1j*pX)
    if hN%2 == 1:
        Y[hN:] = mX[-1:1:-1] * np.exp(-1j*pX[-1:1:-1])
    else:
        Y[hN:] = mX[:0:-1] * np.exp(-1j*pX[:0:-1])
        
    fftbuffer = np.real(ifft(Y));                        # compute inverse FFT
    if hM2%2 == 1:
        y[:hM1] = fftbuffer[-hM1:]                       # undo zero-phase window
        y[hM1:] = fftbuffer[:hM2]
    else: # hM1 == hM2
        y[:hM2] = fftbuffer[-hM2:]                       # undo zero-phase window
        y[hM2:] = fftbuffer[:hM1]
        
    return y

def istft( mY, pY, Parm ):
    '''
    %% Synthesis of a sound using the short-time Fourier transform
    %% Input:
    %           mY: magnitude spectrogram,
    %           pY: phase spectrogram,
    %         Parm: STFT configuration,
    %       Parm.M: window size,
    %       Parm.N: DFT size (no constraint on power 2!),
    %       Parm.H: hop size,
    %       remain: audio signal between the center of last frame and the end of
    %                   audio signal; for synthesis
    %% Ouput:
    %            y: output sound
    '''
    remain = Parm.remain
    M = Parm.M
    N = Parm.N
    H = Parm.H
    
    # prepare istft looping
    hM1 = int(math.floor((M+1)/2))                                # half analysis window size by rounding
    hM2 = int(math.floor(M/2))                                    # half analysis window size by floor
    numFrames = mY[0,:].size                                      # number of frames
    y = np.zeros(hM2 + 1 + (numFrames-1)*H + remain + hM1)        # initialize output array

    ## run istft
    # Note index diff for odd/even size of analysis window
    pin = list(np.arange(hM2, y.size-hM1, H))
    t = 0
    if hM2%2 == 1:
        for i in pin:
            ytmp = dftSynth(mY[:,t], pY[:,t], M, N)
            y[(i-hM1+1):i+hM2+1] = y[(i-hM1+1):i+hM2+1] + H*ytmp    # overlap-add to generate output sound
            t = t + 1
    else:
        for i in pin:
            ytmp = dftSynth(mY[:,t], pY[:,t], M, N)
            y[(i-hM1):i+hM2] = y[(i-hM1):i+hM2] + H*ytmp    # overlap-add to generate output sound
            t = t + 1

    # delete half of first window and half of the last window which was added in stft
    y = y[hM2:-hM1+1]
    
    return y

def peakDetection( mXdB, Parm ):
    '''
    % Detect spectral peak locations
    % mXdB: magnitude spectrum in dB
    % t: MagLevel: 1-64
    % returns mask of peak locations
    '''
    ploc = np.zeros((Parm.numBins,Parm.numFrames))

    for n in np.arange(Parm.numFrames):
        # potential location of peak
        next_minor = np.where(mXdB[1:-1,n]>mXdB[2:,n], 1, 0)      # locations higher than the next one
        prev_minor = np.where(mXdB[1:-1,n]>mXdB[:-2,n], 1, 0)     # locations higher than the previous one
        idx = next_minor * prev_minor                             # locations fulfilling the two criteria
        idx = idx.nonzero()[0] + 1                                # add 1 to compensate for previous steps
        ploc[idx,n] = 1
    
        # peak below Mag Level, cancel
        MagLvl = UT.dBToMagLvl(mXdB[idx,n], Parm.mindB, Parm.maxdB)
        cancelrow = np.where(MagLvl<Parm.t, 1, 0)
        ploc[idx[cancelrow.nonzero()[0]],n] = 0
    
    return ploc

def sineSynth( mYdB, pY, ploc, Parm ):
    '''
    %% Synthesis of a sound using the short-time Fourier transform
    %% Input:
    %         mYdB: magnitude spectrogram in cell type
    %           pY: phase spectrogram in cell type
    %         ploc: frequency info - peak location in matrix type
    %         Parm: STFT configuration,
    %       Parm.M: window size,
    %       Parm.N: DFT size (no constraint on power 2!),
    %       Parm.H: hop size,
    %       remain: audio signal between the center of last frame and the end of
    %                   audio signal; for synthesis
    %% Ouput:
    %            y: output sound
    '''
    remain = Parm.remain
    M = Parm.M
    N = Parm.N
    H = Parm.H

    ## prepare synthesis window
    Ns = 1024                                                     # FFT size for synthesis (eve)
    hNs = 512                                                     # half synthesis window size
    sw = np.zeros(Ns)                                             # initialize synthesis window
    ow = sg.get_window('triang', 2*H)                             # overlapping window    
    sw[hNs-H:hNs+H] = ow                                          # add triangular window
    bh = sg.get_window('blackmanharris', Ns)                      # synthesis window            
    bh = bh / sum(bh)                                             # normalized blackmanharris window                          
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]           # normalize synthesis window

    ## prepare sineSynth looping
    hM1 = int(math.floor((M+1)/2))                                # half analysis window size by rounding
    hM2 = int(math.floor(M/2))                                    # half analysis window size by floor
    numFrames = Parm.numFrames;                                   # number of frames
    y = np.zeros(hM2 + 1 + (numFrames-1)*H + remain + hM1)        # initialize output array

    ## run sineSynth
    pin = list(np.arange(hM2, y.size-hM1, H))
    t = 0
    for i in pin:
        sploc = (np.where(ploc[:,t]==1,1,0)).nonzero()[0] * Ns/N
        Y = cSineModel.genSpecSines( sploc, mYdB[t], pY[t], Ns )
        ytmp = np.real(fftshift(ifft(Y)))
        y[i-hNs:i+hNs] = y[i-hNs:i+hNs] + sw*ytmp             # overlap-add to generate output sound
        t = t + 1

    ## delete half of first window and half of the last window which was added in stft
    y = y[hM2:-hM1+1]
    
    return y

def PT_Algo_FM_C( mXdB, ploc, vploc, Parm ):
    
    partials = cSineModel.PT_Algo_FM( mXdB, ploc, vploc, Parm )
    
    return partials
