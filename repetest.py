# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:26:45 2018
@author: wangruobai
"""

import numpy as np
import librosa
import time
    

def findpeaks(a, th=0, d=1, n=100):
    l=np.shape(a)[0]
    b=a+0
    i=0
    while i<l-1:
        #if (a[i]<=a[i-1])or(a[i]<a[i+1])or(a[i]<th):
        if (a[i]<np.max(a[max(i-d,0):i+d]))or(a[i]<th):
            k=np.argmax(a[i+1:i+d])
            b[i:i+1+k]=0 # delete all non-peaks
            i=i+1+k
        else:
            b[i+1:i+d]=0
            i=i+d
    c=np.flip(np.argsort(b)) # get the indices of peaks from highest to lowest
    
    p=n
    while (p>1) and (b[c[p-1]]<th):
        p=p-1
    c=c[:p] # truncate
    
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
        if (i%5==0):
            print("Grif %d/%d"%(i,iters))
    return y

def similarity_indices(S, th=0, d=1, n=100):
    
    m = np.shape(S)[0] # Number of frames
    I = []
    for j in range(m): # Loop over the frames
        i = findpeaks(S[:,j], th=th, d=d, n=n) # Find local maxima
             # Minimum peak height, distance, Number of peaks, Peak sorting
        I.append(i); # Similarity indices for frame j
        if (j%100==0):
            print("Sim %d/%d"%(j,m))
            print(i[:10])
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


def repet_sim(x, fs, wp=None, cof=None, th=None, d=None, n=None):
    wp = wp or 512 # Analysis window hop in seconds (audio stationary around 40 milliseconds)
    
    cof = cof or 100 # Cutoff frequency in Hz for the dual high-pass filtering (e.g., singing voice rarely below 100 Hz)
    cof = int(cof*(wp*4-1)/fs+0.999) # Cutoff frequency in frequency bins for the dual high-pass filtering (DC component = bin 0)
    
    t = len(x)
    t0 = time.time()
    # Short-Time Fourier Transform (STFT) of channel i
    V0 = np.abs(librosa.core.stft(x,n_fft=wp*4,win_length=wp*4,hop_length=wp))
    n_fft, n_frame = np.shape(V0)
    print(n_frame, n_fft)
    Eng = np.sqrt(np.sum(V0**2, axis=0))
    assert len(Eng)==n_frame
    mEng = np.mean(Eng)
    #Eng = Eng + mEng/10
    Eng = np.clip(Eng, mEng/10, np.inf)
    V = V0 / Eng
    # Magnitude spectrogram (librosa auto cut out mirrored frequencies)
    
    S = np.matmul(V.T, V)
    S1 = S + 0
    S[:-1,:-1] += S1[1:,1:]
    S[1:,1:] += S1[:-1,:-1]
    S1 = S + 0
    S[:-3,:-3] += S1[3:,3:]
    S[3:,3:] += S1[:-3,:-3]
    S1 = None
    S /= 9
    
    th = th or np.mean(S)
    d = d or int(n_frame**0.5)
    n = n or int(3 * n_frame**0.25)
    T = similarity_indices(S, th=th, d=d, n=n); # Similarity indices for all the frames
    S = None
    t1 = time.time()
    print('Sim  Elapsed Time: %8.3f secs'%(t1-t0))
   
    Mi = repeating_mask(V,T); # Repeating mask for channel i
    V = None
    Mi[1:cof,:] = 1; # High-pass filtering of the (dual) non-repeating foreground
    t2 = time.time()
    print('Sim  Elapsed Time: %8.3f secs'%(t1-t0))
    print('Mask Elapsed Time: %8.3f secs'%(t2-t1))
    yi = griffin_lim(Mi*V0,hop_length=wp)
    accom = yi[0:t] # Truncate to the original mixture length
    zi = griffin_lim((1-Mi)*V0,hop_length=wp)
    voice = zi[0:t]
    t3 = time.time()
    print('Sim  Elapsed Time: %8.3f secs'%(t1-t0))
    print('Mask Elapsed Time: %8.3f secs'%(t2-t1))
    print('Grif Elapsed Time: %8.3f secs'%(t3-t2))
    return voice, accom

if __name__=="__main__":
    x, fs = librosa.core.load('Install x Dream.mp3',sr=16000,mono=True)
    v, a = repet_sim(x, fs, n=None)
    librosa.output.write_wav('install_voice.wav',v,fs,norm=True)
    librosa.output.write_wav('install_accom.wav',a,fs,norm=True)