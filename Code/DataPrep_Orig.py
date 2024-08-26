
import os
import array
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import get_array_type
from scipy.fft import fft, fftfreq
from scipy.io import wavfile


'''找到音频文件名'''
def FindFile(path, filetype):
	file_paths = []
	title = []
	filename = []
	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		if os.path.splitext(file)[1] == filetype:  # 判断文件类型
			filename.append(file)
			file_paths.append(os.path.join(file_path))  # 文件所在目录
			title.append(os.path.splitext(file)[0])  # 文件名称
	return file_paths, title, filename


'''读取音频文件'''
#读wav文件
def readwav(dataset, label, filepaths, type):
	for i in range(len(filepaths)):
		filename = filepaths[i]
		print(i)
		Fs, mt = wavfile.read(filename)
		max = np.max(mt)
		if max == 0:
			max = 1
		mt = mt/max
		time = 10
		if len(mt) > time*Fs:
			mt = mt[0:time*Fs]
		else:
			incre = time*Fs-len(mt)
			mt = np.append(mt, np.zeros(incre, dtype=int).reshape(incre, ), 0)

		'''对音频进行傅里叶变换'''
		yf = fft(mt)
		yf = np.abs(yf)
		xf = fftfreq(len(mt), 1 / Fs)

		# 声音频率在80~1100Hz之间，滤去其他频率
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
		'''
		plt.plot(xf, np.abs(yf))
		plt.show()
		'''
		# 标签分类
		k = np.ones([1, 1], dtype = int)
		if type == 'Gaelic':
			k = 0 * k
		elif type == 'English':
			k = 1 * k
		else:
			k = 2 * k
		dataset = np.append(dataset, yf[:, np.newaxis], axis = 1)
		label = np.append(label, k, axis = 1)
	return dataset, label

#读mp3文件
def readmp3(dataset, label, filepaths, type):
	for i in range(len(filepaths)):
		filename = filepaths[i]
		print(i)
		audio = AudioSegment.from_file(filename)
		if audio.frame_rate != 48000:
			print('Not 48000')
			continue
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
		mt = mt/max
		time = 10
		if len(mt) > time * audio.frame_rate:
			mt = mt[0:time * audio.frame_rate]
		else:
			incre = time * audio.frame_rate - len(mt)
			mt = np.append(mt, np.zeros(incre, dtype=int).reshape(incre, ), 0)

		'''对音频进行傅里叶变换'''
		yf = fft(mt)
		yf = np.abs(yf)
		xf = fftfreq(len(mt), 1 / audio.frame_rate)

		# 声音频率在80~1100Hz之间，滤去其他频率
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
		'''
		plt.plot(xf, np.abs(yf))
		plt.show()
		'''
		# 标签分类
		k = np.ones([1, 1], dtype = int)
		if type == 'Gaelic':
			k = 0 * k
		elif type == 'English':
			k = 1 * k
		else:
			k = 2 * k
		dataset = np.append(dataset, yf[:, np.newaxis], axis=1)
		label = np.append(label, k, axis=1)
	return dataset, label


'''添加训练数据集'''
train_x = np.zeros([10200, 0])
train_y = np.zeros([1, 0], dtype = int)

# path = "C:\\Users\\18220\\OneDrive - University of Glasgow\\UoGScientificResearch\\dataset\\English"
path = "D:\\dataset\\English_train"
filetype = '.wav'
type = 'English'
filepaths, _, _, = FindFile(path, filetype)
train_x, train_y = readwav(train_x, train_y, filepaths, type)

# path = "C:\\Users\\18220\\OneDrive - University of Glasgow\\UoGScientificResearch\\dataset\\Gaelic"
path = "D:\\dataset\\Gaelic_train"
# filetype = '.wav'
filetype = '.mp3'
type = 'Gaelic'
filepaths, _, _, = FindFile(path, filetype)
train_x, train_y = readmp3(train_x, train_y, filepaths, type)

# 将训练集打乱
temp = np.concatenate((train_x, train_y))
temp = np.transpose(temp)
np.random.shuffle(temp)
temp = np.transpose(temp)
train_x = temp[:-1, :]
train_y = temp[-1, :][np.newaxis, :]

# 将部分训练集纳入测试集
test_x = np.zeros([10200, 0])
test_y = np.zeros([1, 0], dtype = int)
_, num = train_x.shape
trainratio = 0.8
num_test = int(np.floor(num*trainratio))
test_x = np.append(test_x, train_x[:, num_test:], axis=1)
test_y = np.append(test_y, train_y[:, num_test:], axis=1)
train_x = train_x[:, 0:num_test]
train_y = train_y[:, 0:num_test]


'''添加额外的测试数据集'''
'''
path = ""
filetype = '.wav'
type = 'English'
filepaths, _, _, = FindFile(path, filetype)
test_x, test_y = readwav(test_x, test_y, filepaths, type)

path = "D:\\dataset\\test_Gaelic"
filetype = '.mp3'
type = 'Gaelic'
filepaths, _, _, = FindFile(path, filetype)
test_x, test_y = readmp3(test_x, test_y, filepaths, type)
'''


train_y = train_y.astype(np.int16)
test_y = test_y.astype(np.int16)

'''储存数据集'''
np.savez("dataset.npz", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
