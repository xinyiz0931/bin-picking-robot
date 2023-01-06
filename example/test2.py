import numpy as np
# import matplotlib.pyplot as plt
# from scipy.misc import electrocardiogram
# from scipy.signal import find_peaks
# x = electrocardiogram()[2000:4000]
# print(x.shape)
# peaks, _ = find_peaks(x, height=0)
# plt.plot(x)
# print(peaks.shape)
# plt.plot(peaks, x[peaks], "x")
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.show()
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
# rng = np.random.default_rng()
# x = np.linspace(-3, 3, 50)
# y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
# plt.plot(x, y, 'ro', ms=5)

# spl = UnivariateSpline(x, y)
# xs = np.linspace(-3, 3, 1000)
# plt.plot(xs, spl(xs), 'g', lw=3)

# spl.set_smoothing_factor(0.5)
# plt.plot(xs, spl(xs), 'b', lw=3)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
mu, sigma = 0, 500
x = np.arange(1, 100, 0.1)  # x axis
z = np.random.normal(mu, sigma, len(x))  # noise
y = x ** 2 + z # data
from scipy.signal import savgol_filter
w = savgol_filter(y, 51, 2)
plt.plot(x, y, linewidth=2, linestyle="-", c='r')  # it include some noise
plt.plot(x, w, 'b')  # high frequency noise removed

plt.show()