import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d

x = np.array([0.5,0.51,0.54])
y = np.array([0.4,0.42, 0.3])
tck = interpolate.splrep(x, y, s=0, k=2)
xnew = np.arange(0.5, 0.54, 0.001)
print(xnew)
ynew = interpolate.splev(xnew, tck, der=0)
plt.figure()
plt.plot(x, y, 'x', xnew, ynew, x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.xlim([0.4, 0.6])
plt.title('Cubic-spline interpolation')
plt.show()

