import numpy
import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct

sample_rate,signal=scipy.io.wavfile.read("/home/hlab/Downloads/OSR_us_000_0010_8k.wav")

#读取前3.5s 的数据
signal=signal[0:int(3.5*sample_rate)]
print("raw signal input: ", signal.shape)

#预先处理
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

plt.plot(signal)
plt.plot(emphasized_signal)
plt.show()

