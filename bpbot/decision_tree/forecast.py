import random
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return 1/np.exp(b+x)

# def func(x):
#     return np.exp(-x)+2

def deriv_func(x):
    return -np.exp(-x)

def asymptote():
    alpha = 1
    x = 1e-4
    y1 = func(x)
    i = 0
    while (True):
        i += 1
        deriv = deriv_func(x)
        # x1 = x1 - alpha * deriv1
        x = x - alpha * deriv
        y2 = func(x)
        print(i, x, deriv, y2)
        if deriv < 1e-4 and deriv > -1e-4:
            return x, y2
        # if i > 10: break

def calc_gradient(l):
    g = []
    size = len(l)
    for i in range(size-1):
        # g.append(l[size-1-i]-l[size-2-i])
        g.append(l[i+1]-l[i])
    return np.array(g)
        
import matplotlib.pyplot as plt
y = [0.5, 0.6, 0.44, 0.54]
# y = [1,0.5,0.25,0.1]

i = 0
ini_noise = 0.05
N = 10
while (i<N):
    print("---------- Iteration", i, "---------")
    x = np.arange(0, len(y), 1.0)
    popt, pcov = curve_fit(func, x, y)
    x_ = np.arange(0, len(y)+1, 1.0)
    y_ = func(x_, *popt)
    y_deriv = deriv_func(x_[-1])
    print(i, y_deriv, y_[-1])
    
    if y_deriv < 1e-6 and y_deriv > -1e-6:
        print("Terminate! ")
        break
    print("Fit curve: ", y, end='')
    noise = ini_noise * (1 - i/N)
    y = np.append(y, 0.53333+random.uniform(-noise, noise))
    # y = np.append(y, [0.5+random.uniform(-noise,noise), np.mean(y_[-2])])
    print(' |', y[-2:])
    i+=1

plt.plot(y)
plt.ylim([0,1])

plt.show()

# while (i < 5):
#     print("------- Iteration ", i, " --------")
#     gy = calc_gradient(y)
#     gx = np.arange(0, len(gy), 1.0)
#     popt, pcov = curve_fit(func, gx, gy)
#     gx_ = np.arange(0, len(gy)+1, 1.0)
#     gy_ = func(gx_, *popt) 
#     print("Cauculate new gradient: ", gy_)
#     noise = ini_noise * (1 - i/N)
#     y.append(np.mean(y[-1])+np.mean(gy[-1]))
#     y.append(0.5+random.uniform(-noise,noise))
#     print("Add a new point: ", y[-2:])
#     i += 1
#     break

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.set_title("Origin")
# ax1.plot(y, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# ax2 = fig.add_subplot(212)
# ax2.plot(gy, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# ax2.plot(gy, 'o', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# # ax2.scatter(gx, gy, label='data')
# ax2.set_title("Gradient")
# plt.tight_layout()
# plt.show()

