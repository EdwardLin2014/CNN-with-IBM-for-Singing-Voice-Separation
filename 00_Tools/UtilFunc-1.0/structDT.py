from scipy.signal import get_window

class Param:
    def __init__(self):
        self.M = 2048                                   # Window Size, 46.44ms
        self.window = get_window('hann', self.M)        # Window in Vector Form
        self.N = 8192                                   # Analysis FFT Size, 185.76ms
        self.H = 512                                    # Hop Size, 11.61ms
        self.fs = 44100                                 # Sampling Rate, 44.10K Hz
        self.t = 1                                      # Dicard Peaks below Mag level t
        self.remain = 1
        self.numFrames = 0
        self.numBins = 0
        self.mindB = 0
        self.maxdB = 0
        self.binFreq = 1
        self.freqDevSlope = 0.01
        self.freqDevOffset = 30
        self.MagCond = 4
        self.minPartialLength = 4

class Signal:
    def __init__(self):
        self.x = 0
        self.xL = 0
        self.xR = 0
        self.y = 0
        self.IBMy = 0
        self.IBMPeaky = 0
        self.X = 0
        self.mX = 0
        self.mXL = 0
        self.mXR = 0
        self.mXdB = 0
        self.pX = 0
        self.IBM = 0
        self.IRM = 0
        self.ploc = 0
        self.Partials = 0
        self.PMask = 0