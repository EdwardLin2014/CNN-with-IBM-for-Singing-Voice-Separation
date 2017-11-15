import numpy as np
from os import walk

def iKalaWavFileNames(DatabaseDirStr):
    '''
    Output 
        FileDirs = cell(252,1);
        137 Verse = cell(1:137,1);
        115 Chorus = cell(138:252,1);
    '''
    ## Function Body
    iKala = []
    for (dirpath, dirnames, filenames) in walk(DatabaseDirStr):
        iKala.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(iKala):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    
    FilesDirs = ["" for x in range(numFiles)]
    l = -1;
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        wavname = iKala[i]
        if wavname[-10:] == '_verse.wav':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname
    for i in np.arange(startIdx+1,numFiles+startIdx):
        wavname = iKala[i]
        if wavname[-11:] == '_chorus.wav':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname

    return FilesDirs

def iKalaPitchLabelFileNames(DatabaseDirStr):
    '''
    Output 
        FileDirs = cell(252,1);
        137 Verse = cell(1:137,1);
        115 Chorus = cell(138:252,1);
    '''
    ## Function Body
    iKala = []
    for (dirpath, dirnames, filenames) in walk(DatabaseDirStr):
        iKala.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(iKala):
        if len(filename) > 3:
            if filename[-3:] == '.pv':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    
    FilesDirs = ["" for x in range(numFiles)]
    l = -1;
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        wavname = iKala[i]
        if wavname[-9:] == '_verse.pv':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname
    for i in np.arange(startIdx+1,numFiles+startIdx):
        wavname = iKala[i]
        if wavname[-10:] == '_chorus.pv':
            l += 1
            FilesDirs[l] = DatabaseDirStr + wavname

    return FilesDirs
    
def iKalaPitchMask( PitchFileNames,numMusics ):
    PitchMask = np.zeros((numMusics,937),float);
    
    for n in np.arange(numMusics):
        with open(PitchFileNames[n]) as f:
            i = 0
            for line in f:
                PitchMask[n,i] = line
                i += 1
    PitchMask[PitchMask>1] = 1;
    return PitchMask

def ccMixterWavFileNames(DatabaseDirStr, isLeft):

    ## Function Body
    if isLeft:
        VoiceDirStr = DatabaseDirStr + 'Voice_L/'
        SongDirStr = DatabaseDirStr + 'Bgm_L/'
    else:
        VoiceDirStr = DatabaseDirStr + 'Voice_R/'
        SongDirStr = DatabaseDirStr + 'Bgm_R/'
    # Voice
    ccMixterVoice = []
    for (dirpath, dirnames, filenames) in walk(VoiceDirStr):
        ccMixterVoice.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(ccMixterVoice):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    VoiceFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        VoiceFilesName[l] = VoiceDirStr + ccMixterVoice[i]
        l += 1
    # Song
    ccMixterSong = []
    for (dirpath, dirnames, filenames) in walk(SongDirStr):
        ccMixterSong.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(ccMixterSong):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    SongFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        SongFilesName[l] = SongDirStr + ccMixterSong[i]
        l += 1

    return VoiceFilesName, SongFilesName

def DSD100WavFileNames(DatabaseDirStr):
    ## LMix
    LMixDir = DatabaseDirStr + 'LeftChannel/Mix/'    
    LMix = []
    for (dirpath, dirnames, filenames) in walk(LMixDir):
        LMix.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(LMix):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    LMixFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        LMixFilesName[l] = LMixDir + LMix[i]
        l += 1
    
    ## LVoice
    LVoiceDir = DatabaseDirStr + 'LeftChannel/Voice/'
    LVoice = []
    for (dirpath, dirnames, filenames) in walk(LVoiceDir):
        LVoice.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(LVoice):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    LVoiceFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        LVoiceFilesName[l] = LVoiceDir + LVoice[i]
        l += 1

    ## LSong
    LSongDir = DatabaseDirStr + 'LeftChannel/Song/'
    LSong = []
    for (dirpath, dirnames, filenames) in walk(LSongDir):
        LSong.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(LSong):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    LSongFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        LSongFilesName[l] = LSongDir + LSong[i]
        l += 1

    ## RMix
    RMixDir = DatabaseDirStr + 'RightChannel/Mix/'
    RMix = []
    for (dirpath, dirnames, filenames) in walk(RMixDir):
        RMix.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(RMix):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    RMixFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        RMixFilesName[l] = RMixDir + RMix[i]
        l += 1

    # RVoice
    RVoiceDir = DatabaseDirStr + 'RightChannel/Voice/'
    RVoiceDir = DatabaseDirStr + 'LeftChannel/Voice/'
    RVoice = []
    for (dirpath, dirnames, filenames) in walk(RVoiceDir):
        RVoice.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(RVoice):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    RVoiceFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        RVoiceFilesName[l] = RVoiceDir + RVoice[i]
        l += 1
    
    ## RSong
    RSongDir = DatabaseDirStr + 'RightChannel/Song/'
    RSong = []
    for (dirpath, dirnames, filenames) in walk(RSongDir):
        RSong.extend(filenames)
        break
    numFiles = 0;
    startIdx = -1;
    for i, filename in enumerate(RSong):
        if len(filename) > 3:
            if filename[-4:] == '.wav':
                numFiles += 1
            else:
                startIdx = i
        else:
            startIdx = i
    RSongFilesName = ["" for x in range(numFiles)]
    l = 0
    for i in np.arange(startIdx+1,numFiles+startIdx+1):
        RSongFilesName[l] = RSongDir + RSong[i]
        l += 1
        
    return LMixFilesName, LVoiceFilesName, LSongFilesName, RMixFilesName, RVoiceFilesName, RSongFilesName

def DSD100RawNames(DatabaseDirStr):
    '''
        Only return the Music Names
        Path is needed to be created on the running time
    '''
    TestDir = DatabaseDirStr + 'Test/'
    TestNames = []
    for (dirpath, dirnames, filenames) in walk(TestDir):
        if dirnames != '.' or dirnames != '..' or dirnames != '.DS_Store':
            TestNames.extend(dirnames)
        break
    DevDir = DatabaseDirStr + 'Dev/'
    DevNames = []
    for (dirpath, dirnames, filenames) in walk(DevDir):
        if dirnames != '.' or dirnames != '..' or dirnames != '.DS_Store':
            DevNames.extend(dirnames)
        break
    return TestNames, DevNames




