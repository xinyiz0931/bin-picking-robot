import numpy as np
def func(x):
    return np.exp(-x)+2

def deriv_x(x):
    return -np.exp(-x)

def asymptote():
    alpha = 1
    x = 1e-4
    y1 = func(x)
    i = 0
    l = []
    while (True):
        i += 1
        deriv = deriv_x(x)
        # x1 = x1 - alpha * deriv1
        x = x - alpha * deriv
        y2 = func(x)
        print(i, x, deriv, y2)
        l.append(y2)
        if deriv < 1e-4 and deriv > -1e-4:
            return x, y2, l
        # if i > 10: break

import matplotlib.pyplot as plt
y = [0.5, 0.301, 0.528, 0.606, 0.44, 0.48, 0.54]
_,_,l = asymptote()
import matplotlib.pyplot as plt
# plt.ylim([-1,1])
# plt.plot([0,1,2,3,4,5,6], [0.5]*7, alpha=0.3)
# plt.plot(y)
# plt.show()

plt.ylim([2,4])
plt.plot(range(10), [3,2.9,2.9,2.9,2.8,2.7,2.7,2.7,2.6,2.6])
plt.show()