import numpy as np

def prepareSineSynth( X, ploc, Parm ):
    '''
    %%
    %   X: numBins*numFrames matrix, e.g. mX, pX
    %   ploc: peak location
    %   Parm: system configuration
    %   retrun cX: cell(1,numFrames)
    '''
    numFrames = Parm.numFrames
    cX = [[]] * numFrames

    for n in np.arange(numFrames):
        cX[n] = X[(ploc[:,n]==1),n]

    return cX

def PartialsToBinaryMask( Partials, Parm ):
    '''
    %% Transform Partials into a binary mask
    %   Partials: struct type
    %   Parm: System configuration
    %   return ploc: peak location as a numBins*numFrames matrix
    '''
    ploc = np.zeros((Parm.numBins, Parm.numFrames))

    numPartials = len(Partials)
    for i in np.arange(numPartials):
        Partial = Partials[i]
        for j in np.arange(Partial.size):
            ploc[Partial.freqIdx[j],Partial.period[0]+(j-1)] = 1
    
    return ploc
