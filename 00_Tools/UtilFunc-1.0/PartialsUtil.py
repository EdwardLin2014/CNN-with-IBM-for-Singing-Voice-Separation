import numpy as np

def ClassifyPartials( Partials ):

    VoicePartials = []
    SongPartials = []

    numPartials = len(Partials)
    for idx in np.arange(numPartials):
        Partial = Partials[idx]
        
        numVoice = np.count_nonzero(Partial.type)
        numSong = np.count_nonzero(Partial.type==0)        
    
        if numVoice > numSong:
            VoicePartials.append(Partials[idx])
        else:
            SongPartials.append(Partials[idx])

    return VoicePartials, SongPartials