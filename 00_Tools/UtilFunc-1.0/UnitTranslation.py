import numpy as np

def dBToMag( dB ):
    Mag = np.power(10, np.divide(dB,20));
    # if zeros add epsilon(eps) to handle log
    # The smallest representable number such that 1.0 + eps != 1.0.
    if isinstance(Mag, float):
        if Mag<np.finfo(float).eps:
            Mag = np.finfo(float).eps
    else:
        Mag[Mag<np.finfo(float).eps] = np.finfo(float).eps

    return Mag


def dBToMagLvl( dB, mindB, maxdB ):
    '''
        Divide up the magnitude spectrogram into Magnitude Level
        Based on "How Color Axis Scaling Works" in Matlab
        https://www.mathworks.com/help/matlab/ref/caxis.html
    '''
    Tlvl = 64;     # Default Total Level in Matlab

    MagLevel = np.fix((dB-mindB)/(maxdB-mindB)*Tlvl)+1;
    #Clamp values outside the range [1 m]    
    if isinstance(MagLevel, float):
        if MagLevel < 1:
            MagLevel = 1
        if MagLevel > Tlvl:
            MagLevel = Tlvl
    else:
        MagLevel[MagLevel<1] = 1;
        MagLevel[MagLevel>Tlvl] = Tlvl; 

    return MagLevel

def FBinToHz( FBin, Parm ):
    #       FBin: [0 ... N-1]
    #    Parm.fs: sampling rate
    #     Parm.N: DFT Size
    MaxFBin = np.floor(Parm.N/2)+1;

    if isinstance(FBin, float):
        if FBin > MaxFBin:
            FBin = MaxFBin
        elif FBin < 1:
            FBin = 1
    else:
        FBin[FBin > MaxFBin] = MaxFBin;
        FBin[FBin < 1] = 1;
    
    Hz = Parm.fs/Parm.N * (FBin-1);
    return Hz

def HzToFBin( Hz, Parm ):
    '''
              Hz: 0 ... Parm.fs
         Parm.fs: sampling rate
          Parm.N: DFT Size
    '''
    MaxFBin = np.floor(Parm.N/2)+1;
    FBin = round(Parm.N/Parm.fs * Hz) + 1;

    if isinstance(FBin, float):
        if FBin > MaxFBin:
            FBin = MaxFBin
        elif FBin < 1:
            FBin = 1
    else:
        FBin[FBin>MaxFBin] = MaxFBin;
        FBin[FBin<1] = 1;

    return FBin

def MagTodB( Mag ):
    ## if zeros add epsilon(eps) to handle log
    # The smallest representable number such that 1.0 + eps != 1.0.
    # This code should not be executed?
    if isinstance(Mag, float):
        if Mag<np.finfo(float).eps:
            Mag = np.finfo(float).eps 
    else:
        Mag[Mag<np.finfo(float).eps] = np.finfo(float).eps;
    
    dB = 20*np.log10(Mag);
    return dB

