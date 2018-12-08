 # -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:26:45 2018
@author: wangruobai
"""

import numpy as np
import librosa
    

def findpeaks(a, sim):
    th=sim[0]
    d=sim[1]
    n=sim[2]
    l=np.shape(a)[0]
    b=[]
    c=[]
    for i in range(1,l-1):
        if (a[i]>a[i-1])and(a[i]>=a[i+1])and(a[i]>=th):
            b.append(a[i])
            c.append(i)
    b=np.array(b)
    c=np.array(c)
    b=np.flipud(np.argsort(b))
    c=c[b]
    l=np.shape(c)[0]
    p=0
    while ((p<l)and(p<n)):
        f=True
        for i in range(p):
            if (np.abs(c[p]-c[i])<d):
                f=False
                break
        if (f):
            p=p+1
        else:
            c=np.delete(c,p)
            l=l-1
    return c

def griffin_lim(stftm, hop_length=0, iters=50, center=True):
    n_fft = (np.shape(stftm)[0]-1)*2
    if (hop_length==0):
        hop_length=n_fft//4
    n_window = np.shape(stftm)[1]
    yshape = hop_length * (n_window-1) + (0 if center else n_fft)
    y = np.random.random(yshape)
    for i in range(iters):
        stftx = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, center=center)
        stftx = stftm * stftx / (np.abs(stftx) + 0.0001)
        y = librosa.core.istft(stftx, hop_length=hop_length, center=center)
    return y

def similarity_indices(S,sim):
    
    m = np.shape(S)[0] # Number of frames
    I = []
    for j in range(m): # Loop over the frames
        i = findpeaks(S[:,j], sim) # Find local maxima
             # Minimum peak height, distance, Number of peaks, Peak sorting
        I.append(i); # Similarity indices for frame j
        if (j%100==0):
            print("Sim %d/%d"%(j,m))
    return I
    
def repeating_mask(V,I):
    eps=1e-8
    n,m = np.shape(V) # Number of frequency bins and time frames
    W = np.zeros((n,m))
    for j in range(m): # Loop over the frames
        i = I[j] # Similarities indices for frame j (i(1) = j)
        if (len(i)==0):
            continue
        W[:,j] = np.median(V[:,i],axis=1) # Median of the similar frames for frame j
        if (j%100==0):
            print("Mask %d/%d"%(j,m))
    W = np.minimum(V,W); # For every time-frequency bins, we must have W <= V
    M = (W+eps)/(V+eps); # Normalize W by V
    return M


def repet_sim(x, fs, par=[0,1,100]):
    leng = 0.04 # Analysis window length in seconds (audio stationary around 40 milliseconds)
    N = 2**int(np.log2(leng*fs-0.001)+1) # Analysis window length in seconds (audio stationary around 40 milliseconds)
    stp = N//2 # Analysis step length (N/2 for constant overlap-add)
    
    cof = 100 # Cutoff frequency in Hz for the dual high-pass filtering (e.g., singing voice rarely below 100 Hz)
    cof = int(cof*(N-1)/fs+0.999) # Cutoff frequency in frequency bins for the dual high-pass filtering (DC component = bin 0)
    
    t = np.shape(x)[0] # Number of samples and channels
    X = librosa.core.stft(x,n_fft=N,hop_length=stp) # Short-Time Fourier Transform (STFT) of channel i
    V = np.abs(X) # Magnitude spectrogram (librosa auto cut out mirrored frequencies)
    
    S = np.corrcoef(V.T)
    par[1] = round(par[1]*fs/stp); # Distance in time frames
    S = similarity_indices(S,par); # Similarity indices for all the frames
    
    Mi = repeating_mask(V,S); # Repeating mask for channel i
    Mi[1:cof,:] = 1; # High-pass filtering of the (dual) non-repeating foreground
    yi = griffin_lim(Mi*V,hop_length=stp)
    accom = yi[0:t] # Truncate to the original mixture length
    zi = griffin_lim((1-Mi)*V,hop_length=stp)
    voice = zi[0:t]
    return voice, accom

if __name__=="__main__":
    x, fs = librosa.core.load('D:\\codes\\py\\strike.wav',sr=None)
    v, a = repet_sim(x,fs)
    librosa.output.write_wav('D:\\codes\\py\\strike_voice.wav',v,fs,norm=True)
    librosa.output.write_wav('D:\\codes\\py\\strike_accom.wav',a,fs,norm=True)
