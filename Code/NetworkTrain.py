import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle


def conv(input, kernel, type):
    kernel = np.rot90(np.rot90(kernel))
    output_size = (len(input)+2*len(kernel)-2)
    convlen = len(input) + len(kernel) - 1
    res = np.zeros([convlen, convlen], np.float32)
    Newinput = np.zeros([output_size, output_size], np.float32)
    Newinput[(len(kernel) - 1):(len(kernel) + len(input) - 1), (len(kernel) - 1):(len(kernel) + len(input) - 1)] = input
    for i in range(convlen):
        for j in range(convlen):
            res[i][j] = compute_conv(Newinput, kernel, i, j)
    if type == 'full':
        output = res
    elif type == 'same':
        begin = int((len(kernel)-1)/2)
        end = int((len(kernel)-1)/2+len(input))
        output = res[begin:end, begin:end]
    else:
        output = res[(len(kernel)-1):len(input), (len(kernel)-1):len(input)]
    return output


def compute_conv(Newinput, kernel, i, j):
    res = 0
    for kk in range(len(kernel)):
        for k in range(len(kernel)):
            # print(input[i+kk][j+k])
            res += Newinput[i+kk][j+k]*kernel[kk][k]
    return res


def ConvolutionLayer(BP, X, kernel_size, ActiFunc, alpha, e_n):
    # ConvolutionLayer

    # BP
    # X
    # kernel_size
    # ActiFunc
    # alpha
    # e_n

    # Y
    # e_n_1
    # dWn

    (X_row, none, Channel_num) = np.shape(X)
    (Kernel_row, none, Kernel_num) = np.shape(kernel_size)


    NewKernelSize = np.expand_dims(kernel_size, axis=2).repeat(Channel_num, axis=2)

    FM_size = X_row - Kernel_row + 1  # valid
    V = np.zeros([FM_size, FM_size, 1, Kernel_num])
    for j in range(Kernel_num):
        V[:, :, 1, j] = conv(X, np.rot90(np.rot90(NewKernelSize[:, :, :, j])), 'valid')
    V = np.squeeze(V)

    # activation function
    if ActiFunc == 'ReLU':
        Y = np.max(0, V)
    elif ActiFunc == 'Sigmoid':
        Y = Sigmoid(V)
    else:
        Y = Sigmoid(V)

    e_n_1 = []
    dWn = []
    if BP == 1:  # back forward
        if ActiFunc == 'ReLU':
            delta = np.dot(np.maximum(Y, 0), e_n)
        elif ActiFunc == 'Sigmoid':
            delta = np.dot(np.dot(Y, (1 - Y)), e_n)
        else:
            delta = np.dot(np.dot(Y, (1 - Y)), e_n)


        delta = np.swapaxes(delta, 2, 3)
        e_n_1_size = FM_size + Kernel_row - 1
        e_n_1 = np.zeros([e_n_1_size, e_n_1_size, Channel_num])
        for j in range(Channel_num):
            for i in range(Kernel_num):
                e_n_1[:, :, j] = e_n_1[:, :, j] + conv(delta[:, :, 1, i], NewKernelSize[:, :, 1, i], 'full')

        # weights
        dWn = np.zeros(np.shape(NewKernelSize))
        for i in range(Kernel_num):
            dWn[:, :, :, i] = alpha * conv(X, np.rot90(np.rot90(delta[:, :, :, i])), 'valid')
        dWn = dWn[:, :, 1, :]
        dWn = np.squeeze(dWn)
    return [Y, e_n_1, dWn]


def PoolingLayer(BP, X, Pooling_Method, e_n):
    # PoolingLayer
    # BP
    # X
    # Pooling_Method
    # e_n
    # e_n_1

    # 正向传播
    if Pooling_Method == 'average':
        V = (X[0::2, 0::2]+X[1::2, 0::2]+X[0::2, 1::2]+X[1::2, 1::2])/4
    Y = V

    e_n_1 = np.zeros(np.shape(X))
    if BP:
        ysize = np.shape(Y)
        e_n = e_n.reshape(ysize)
        kernel_num = ysize[2]
        if Pooling_Method == 'average':
            Average_kernel = np.ones([2, 2])*1/4
            for k in range(kernel_num):
                e_n_1[:, :, k][:, :, np.newaxis] = np.kron(e_n[:, :, k][:, :, np.newaxis], Average_kernel)

    return [Y, e_n_1]


def FullConnLayer(BP, X, W, ActiFunc, alpha, e_n, CostFunc, LastLayer, lamda):
    # FullConnLayer
    # BP
    # X
    # W
    # ActiFunc
    # alpha
    # e_n
    # CostFunc
    # LastLayer
    # lamda
    # Y
    # e_n_1
    # dWn


    if X.ndim == 3:
        [X_row, X_column, X_num] = np.shape(X)
        X = np.reshape(X_row*X_column*X_num, 1)
    V = np.dot(W, X)

    # activation function
    if ActiFunc == 'ReLU':
        Y = np.max(0, V)
    elif ActiFunc == 'Sigmoid':
        Y = Sigmoid(V)
    elif ActiFunc == 'Softmax':
        Y = Softmax(V)

    e_n_1 = []
    dWn = []
    if BP:
        if LastLayer:
            if CostFunc == 'CrossEntro':
                delta = e_n
            elif CostFunc == 'ErrorAverage':
                delta = Y * (1 - Y) * e_n
            e_n_1 = np.dot(np.transpose(W), delta)
            dWn = alpha * np.dot(delta, np.transpose(X)) - alpha * lamda * W
        else:
            if ActiFunc == 'ReLU':
                delta = np.dot(np.maximum(Y, 0), e_n)
            elif ActiFunc == 'Sigmoid':
                delta = Y * (1 - Y) * e_n
            e_n_1 = np.dot(np.transpose(W), delta)
            dWn = alpha * np.dot(delta, np.transpose(X))
    return [Y, e_n_1, dWn]


def Softmax(x):
    ex = np.exp(x)
    y = ex/sum(ex)
    return y

'''
def Sigmoid(x):
    y = 1./(1+np.exp(-x))
    return y
'''

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


def Accuracy(d_testResult, test_y):
    # Accuracy

    # d_testResult
    # test_y

    # prct_acc
    num = test_y.shape[1]
    acc = 0  #
    for k in range(num):
        if d_testResult[k] == int(test_y[0, k]):
            acc += 1
    prct_acc = acc/num

    return prct_acc


def DNNtrain(train_x, train_y, test_x, test_y, Network, alpha, CostFunc, weight_update_strategy, SmaBat_size, beta,
             lamda, Epoch, SamplePiont, learning_rate_decay):
    # DNNtrain
    # train_x
    # train_y
    # test_x
    # test_y
    # NetworkStruc
    # alpha
    # CostFunc
    # weight_update_strategy
    # SmaBat_size
    # beta
    # lamda
    # Epoch
    # SamplePiont
    # learning_rate_decay
    # NetworkStruc
    nTrainSample = 0
    batch_size = 0
    if train_x.ndim == 3:
        nTrainSample = train_x.shape[2]
    elif train_x.ndim == 2:
        nTrainSample = train_x.shape[1]

    if weight_update_strategy == "SGD":
        batch_size = 1
    elif weight_update_strategy == "Batch":
        batch_size = nTrainSample
    elif weight_update_strategy == "SmaBat":
        batch_size = SmaBat_size

    nX = math.floor(Epoch * nTrainSample / SamplePiont)
    Error_acummu = np.zeros(nX)
    error_acummu = 0.0
    nAcc = np.zeros(Epoch)
    nlayers = len(Network)
    for i_epoch in range(Epoch):
        for Sample in range(nTrainSample):
            # training a sample
            y = []
            if train_x.ndim == 3:
                y.append(train_x[..., ..., Sample][:, :, np.newaxis])
            elif train_x.ndim == 2:
                y.append(train_x[..., Sample][:, np.newaxis])
            error = [0 for t in range(nlayers)]


            for k1 in range(nlayers):
                BP = 0
                if Network[k1].name == "Convolutionlayer":
                    result = ConvolutionLayer(BP, y[k1], Network[k1].kernel, Network[k1].ActiFunc, alpha, error[k1])
                    Y = result[0]
                    y.append(Y)
                elif Network[k1].name == "Poolinglayer":
                    result = PoolingLayer(BP, y[k1], Network[k1].Pooling_Method, error[k1])
                    Y = result[0]
                    y.append(Y)
                elif Network[k1].name == "FullConnectionlayer":
                    result = FullConnLayer(BP, y[k1], Network[k1].weight, Network[k1].ActiFunc, alpha, error[k1],
                                           CostFunc, 0, lamda)
                    Y = result[0]
                    y.append(Y)


            if train_x.ndim == 3:
                error[nlayers-1] = train_y[..., ..., Sample][:, :, np.newaxis] - y[-1]
            elif train_x.ndim == 2:
                error[nlayers-1] = train_y[..., Sample][:, np.newaxis] - y[-1]
            error_acummu = error_acummu + float(sum(abs(error[nlayers-1])))
            for k2 in range(nlayers, 0, -1):
                BP = 1
                if Network[k2-1].name == "Convolutionlayer":
                    result = ConvolutionLayer(BP, y[k2-1], Network[k2-1].kernel, Network[k2-1].ActiFunc, alpha,
                                              error[k2-1])
                    Error = result[1]
                    dWn = result[2]
                    if k2 != 1:
                        error[k2-2] = Error
                    Network[k2-1].mmt = beta * Network[k2-1].mmt + dWn
                    Network[k2-1].dW_accum = Network[k2-1].dW_accum + Network[k2-1].mmt
                elif Network[k2-1].name == "Poolinglayer":
                    result = PoolingLayer(BP, y[k2-1], Network[k2-1].Pooling_Method, error[k2-1])
                    Error = result[1]
                    error[k2-2] = Error
                elif Network[k2-1].name == "FullConnectionlayer":
                    if k2 == nlayers:
                        LastLayer = True
                    else:
                        LastLayer = False
                    result = FullConnLayer(BP, y[k2-1], Network[k2-1].weight, Network[k2-1].ActiFunc, alpha,
                                           error[k2-1], CostFunc, LastLayer, lamda)
                    Error = result[1]
                    dWn = result[2]
                    if k2 != 1:
                        error[k2-2] = Error
                    Network[k2-1].mmt = beta * Network[k2-1].mmt + dWn
                    Network[k2-1].dW_accum = Network[k2-1].dW_accum + Network[k2-1].mmt


            if Sample % batch_size == 0:
                for k3 in range(nlayers):
                    if Network[k3].name == "Convolutionlayer":
                        Network[k3].kernel = Network[k3].kernel + Network[k3].dW_accum/batch_size
                        Network[k3].dW_accum = np.zeros(Network[k3].kernel_size, dtype=float)
                    elif Network[k3].name == "FullConnectionlayer":
                        Network[k3].weight = Network[k3].weight + Network[k3].dW_accum/batch_size
                        Network[k3].dW_accum = np.zeros(Network[k3].weight_size, dtype=float)

            SmpNum = Sample + i_epoch*nTrainSample + 1
            if SmpNum % SamplePiont == 0:
                print(SmpNum, 'samples trained, error: %.3f'% error_acummu)
                Error_acummu[int(SmpNum/SamplePiont)-1] = error_acummu
                error_acummu = 0

        alpha = learning_rate_decay * alpha


        prct_acc = DNNtest(test_x, test_y, Network)[1]
        nAcc[i_epoch] = prct_acc
        print(i_epoch+1, 'epoch finished, model accuracy：%.3f'% (prct_acc*100), '%')


        if beta != 0:
            for k4 in range(nlayers):
                if Network[k4].name == "Convolutionlayer":
                    Network[k4].mmt = np.zeros(Network[k4].kernel_size, dtype=float)
                elif Network[k4].name == "FullConnectionlayer":
                    Network[k4].mmt = np.zeros(Network[k4].weight_size, dtype=float)

    plt.xlabel('Samples')
    plt.ylabel('Error')
    x = np.arange(nX)
    x = x*100
    y = Error_acummu
    plt.plot(x, y)
    for i in range(Epoch):
        k = math.floor((i+1) * nTrainSample / SamplePiont)
        x0 = x[k-1]
        y0 = Error_acummu[k-1]
        plt.scatter(x0, y0, s=100)
        plt.annotate('Accuracy=%f' % nAcc[i], xy=(x0, y0), xycoords='data', xytext=(+0, +30),
                     textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc,angleA=90,angleB=0,rad=0"))
    # arc,angleA=180,angleB=0,armA=50,armB=0,rad=5
    # arc3,rad=.2
    plt.title('Error-Sample')
    plt.show()
    return Network


def DNNtest(test_x, test_y, Network):


    if test_x.ndim == 3:
        nTrainSample = test_x.shape[2]
    elif test_x.ndim == 2:
        nTrainSample = test_x.shape[1]
    d_DNNtestResult = []  # 构建测试结果向量
    for Sample in range(nTrainSample):

        n = len(Network)
        y = []

        if test_x.ndim == 3:
            y.append(test_x[:, :, Sample][:, :, np.newaxis])
        elif test_x.ndim == 2:
            y.append(test_x[:, Sample][:, np.newaxis])
        error = []

        for k1 in range(n):
            BP = 0
            if Network[k1].name == "Convolutionlayer":
                result = ConvolutionLayer(BP, y[k1], Network[k1].kernel, Network[k1].ActiFunc, 0, 0)
                Y = result[0]
                y.append(Y)
            elif Network[k1].name == "Poolinglayer":
                result = PoolingLayer(BP, y[k1], Network[k1].Pooling_Method, error[k1])
                Y = result[0]
                y.append(Y)
            elif Network[k1].name == "FullConnectionlayer":
                result = FullConnLayer(BP, y[k1], Network[k1].weight, Network[k1].ActiFunc, 0, 0, 0, 0, 0)
                Y = result[0]
                y.append(Y)

        output = y[n]
        index = np.where(output == np.max(output))
        index = int(index[0])
        d_DNNtestResult.append(index)

    prct_acc = Accuracy(d_DNNtestResult, test_y)
    return [d_DNNtestResult, prct_acc]





r = np.load("dataset.npz")
train_x = r["train_x"]
temptrain_y = r["train_y"]
num1 = np.max(temptrain_y)
_, num2 = temptrain_y.shape
train_y = np.zeros([num1+1, num2], dtype=int)
for k in range(num2):
    T = temptrain_y[-1, k]
    train_y[T, k] = 1
test_x = r["test_x"]
test_y = r["test_y"]






alpha = 0.5
learning_rate_decay = 1
CostFunc = 'CrossEntro'
weight_update_strategy = 'SGD'
SmaBat_size = 0
beta = 0
lamda = 0
Epoch = 3
SamplePiont = 100


class Convolution:
    def __init__(self, sequence, weight_filler, size, ActiFunc):
        self.name = "Convolutionlayer"
        self.sequence = sequence
        self.kernel = 0
        self.kernel_size = size
        self.dW_accum = np.zeros(size, dtype=float)
        self.mmt = np.zeros(size, dtype=float)
        self.ActiFunc = ActiFunc
        if weight_filler == 'NormDist':
            self.kernel = np.random.randn(size[0], size[1])


class Pooling:
    def __init__(self, sequence, Pooling_Method):
        self.name = "Poolinglayer"
        self.sequence = sequence
        self.Pooling_Method = Pooling_Method


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


Network = []
Network.append(FullConnection(1, "StandInitial", [20, 10200], "Sigmoid"))
Network.append(FullConnection(2, "StandInitial", [2, 20], "Sigmoid"))

# training model
Network = DNNtrain(train_x, train_y, test_x, test_y, Network, alpha, CostFunc, weight_update_strategy, SmaBat_size,
                        beta, lamda, Epoch, SamplePiont, learning_rate_decay)
# validation model
d_DNNtestResult = DNNtest(test_x, test_y, Network)[0]

with open('Network.txt','wb') as f:
    pickle.dump(Network, f)
