import _pickle as pickle
import numpy as np
import math
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
from pydub.utils import get_array_type
import array
import os
import random


def readwav(filename):
    Fs, mt = wavfile.read(filename)
    max = np.max(mt)
    if max == 0:
        max = 1
    mt = mt / max
    time = 10
    if len(mt) > time * Fs:
        mt = mt[0:time * Fs]
    else:
        incre = time * Fs - len(mt)
        mt = np.append(mt, np.zeros(incre, dtype=int).reshape(incre, ), 0)

    '''Fourier transform'''
    yf = fft(mt)
    yf = np.abs(yf)
    xf = fftfreq(len(mt), 1 / Fs)

    # filter the frequency of audio file in 80~1100Hz
    FH = 1100
    FL = 80
    a = int(np.argwhere(xf >= FL)[0])
    b = int(np.argwhere(xf > FH)[0]) - 1
    xf = xf[a:b]
    yf = yf[a:b]
    max = np.max(yf)
    if max == 0:
        max = 1
    yf = 0.1 * yf / max
    return yf


def readmp3(filename):
    audio = AudioSegment.from_file(filename)
    sound = AudioSegment.from_mp3(filename)
    left = sound.split_to_mono()[0]
    right = sound.split_to_mono()[1]

    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)
    left_numeric_array = array.array(array_type, left._data)
    right_numeric_array = array.array(array_type, right._data)
    left_channel = np.array(left_numeric_array) / 32768
    right_channel = np.array(right_numeric_array) / 32768

    mt = left_channel + right_channel
    max = np.max(mt)
    if max == 0:
        max = 1
    mt = mt / max
    time = 10
    if len(mt) > time * audio.frame_rate:
        mt = mt[0:time * audio.frame_rate]
    else:
        incre = time * audio.frame_rate - len(mt)
        mt = np.append(mt, np.zeros(incre, dtype=int).reshape(incre, ), 0)

    '''Fourier transform''''
    yf = fft(mt)
    yf = np.abs(yf)
    xf = fftfreq(len(mt), 1 / audio.frame_rate)

    # filter the frequency of audio file in 80~1100Hz
    FH = 1100
    FL = 80
    a = int(np.argwhere(xf >= FL)[0])
    b = int(np.argwhere(xf > FH)[0]) - 1
    xf = xf[a:b]
    yf = yf[a:b]
    max = np.max(yf)
    if max == 0:
        max = 1
    yf = 0.1 * yf / max
    return yf


def FullConnLayer(X, W, ActiFunc):
    V = np.dot(W, X)
    if ActiFunc == 'ReLU':
        Y = np.max(0, V)
    elif ActiFunc == 'Sigmoid':
        Y = Sigmoid(V)
    elif ActiFunc == 'Softmax':
        Y = Softmax(V)
    return Y


def Softmax(x):
    ex = np.exp(x)
    y = ex/sum(ex)
    return y


def Sigmoid(x):
    x_ravel = x.ravel()
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


def DNN(mt, Network):
    n = len(Network)  # n -- hidden layers
    y = []
    y.append(mt)
    error = []  # error{1,n}为第n层的e
    for k1 in range(n):
        Y = FullConnLayer(y[k1], Network[k1].weight, Network[k1].ActiFunc)
        y.append(Y)
    output = y[n]
    index = np.where(output == np.max(output))
    index = int(index[0])
    return index


class FullConnection:
    def __init__(self, sequence, weight_filler, size, ActiFunc):
        self.name = "FullConnectionlayer"
        self.sequence = sequence
        self.weight_size = size
        self.dW_accum = np.zeros(size, dtype=float)
        self.mmt = np.zeros(size, dtype=float)
        self.ActiFunc = ActiFunc
        if weight_filler == 'NormDist':
            self.weight = np.random.randn(size)
        elif weight_filler == 'StandInitial':
            self.weight = (2*np.random.rand(size[0], size[1])-1)/(size[1]/size[0])
        elif weight_filler == 'Xavier':
            self.weight = (2*np.random.rand(size[0], size[1])-1)/math.sqrt((size[1]+size[0])/6)

# import model
fid = open('Network.txt', 'rb')
Network = pickle.load(fid)
fid.close()




# read audio files in folder
time = 0
folderpath = "D:\\dataset\\test"
filelist = os.listdir(folderpath)
totalnum = len(filelist)
random.shuffle(filelist)
for file in filelist:
    filepath = os.path.join(folderpath, file)
    filetype = os.path.splitext(file)[1]
    filename = os.path.splitext(file)[0]
    audiofile = filename + filetype
    if filetype == '.mp3':
        mt = readmp3(filepath)
    elif filetype == '.wav':
        mt = readwav(filepath)
    classresult = DNN(mt, Network)
    if classresult == 0:
        type = 'Gaelic'
        time += 1
    elif classresult == 1:
        type = 'English'
    else:
        type = 'None'
    print('File name:', audiofile, '-- language: ', type)  # 文件名称