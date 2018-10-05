from scipy.io import wavfile
from scipy import hanning
import numpy as np
import os


#https://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python/20409020
def stft(x, fftsize=64, overlap_pct=.5):   
    hop = int(fftsize * (1 - overlap_pct))
    w = hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]

def load_dataset(directory): 
    Y = []
    X = []        
    for f in os.listdir(directory + "\\classic\\"):
        if ".wav" in f:
            stft_data = np.abs(stft(wavfile.read(directory+"\\classic\\"+f)[1]));
            X.append(np.mean(stft_data[0:3000,:],axis=1))
            Y.append(1)
    
         
    for f in os.listdir(directory + "\\modern\\"):
        if ".wav" in f:
            stft_data = np.abs(stft(wavfile.read(directory+"\\modern\\"+f)[1]));
            X.append(np.mean(stft_data[0:3000,:],axis=1))
            Y.append(0)
    
    return np.array(X),np.array(Y)