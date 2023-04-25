import numpy as np
import matplotlib.pyplot as plt
import time
from bpbot.device import DynPickClient, DynPickControl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
dc = DynPickClient()
fs = DynPickControl()
F0_reset = np.array([8151,8223,8487,8287,8102,8322])
sensitivity = np.array([32.800,32.815,32.835,1653.801,1634.816,1636.136])

# def fit_linear(y):
#     def linear(x,b):
#         return b
#     y = np.array(y)
#     x = np.arange(y.shape[0])
#     popt, pcov = curve_fit(linear, x, y)
#     return popt[0]


# f_fail = [1, 0.204, 0.630, 0.746, 0.649, 0.9745, 0.636, 0.192, 0.636, 0.615, 0.624, 0.667, 0.612, 0.639, 0.316, 0.603, 0.408, 0.800, 0.580, 0.444, 0.758]
# new_f_fail = []
# f_fail_opt = []
# fit_linear(f_fail)
# j = 0
# for i, f in enumerate(f_fail):
#     if i < 5:
#         j = i
#         new_f_fail.append(f)
#         f_fail_opt.append(f)
#         continue
#     f_opt = fit_linear(new_f_fail[:j])
#     new_f_fail.append(f_opt)
#     new_f_fail.append(f)
#     f_fail_opt.append(f_opt)
#     j += 2
#     print(i, "->", f_opt)


# plt.plot(f_fail_opt)
# plt.scatter(list(range(len(f_fail))),f_fail,  alpha=0.5)
# plt.yticks(np.arange(-1.5,2,0.25))
# plt.xticks([int(x) for x in range(len(f_fail_opt))])
# plt.show()

# f_stop = [3,2.9,2.9,2.9,2.8,2.7,2.6,2.6,2.6]+[2.5]*11
# plt.plot(f_stop)
# plt.yticks(np.arange(-0.5,4,0.25))
# plt.xticks([int(x) for x in range(len(f_stop))])
# plt.show()
# # def func(x, a, b, c):
# #     return a / (1 + np.exp(- b * (x - c)))
# def func(x,a,b,c):
#     return a/(1+np.exp(-b*x))+c
#     # return a*np.exp(-c/x)
#     # return a*x**b*np.exp(-c/x)
#     # return a*((x/298)**b)*np.exp(-c/(0.008314*x))

# def calc_gradient(l):
#     g = []
#     size = len(l)
#     for i in range(size-1):
#         # g.append(l[size-1-i]-l[size-2-i])
#         g.append(l[i+1]-l[i])
#     return np.array(g)

# def monitoring(max_tm=6, frequeny=20, thld=2, stop=True):
#     _tm = 1/frequeny
#     ft_out = []
#     curr_jnt = []
#     # fz_gradient = []
#     for i in np.arange(0, max_tm, _tm):
#         ft = dc.get()
#         ft = (ft-F0_reset)/sensitivity

#         ft_out.append(ft if ft[2] > 0 else 0)
#         print(ft)
#         time.sleep(_tm)

#     ft_out = np.array(ft_out)

#     fig = plt.figure(figsize=(3,3))
#     ax1 = fig.add_subplot(211)
#     ax1.axhline(y=1.2, color='gold', alpha=.7, linestyle='dashed')
#     ax1.plot(ft_out[:,2], alpha=0.3)
#     ax1.set_yticks(np.arange(-1,1,0.1))
#     ax2 = fig.add_subplot(212)
#     # ax2.plot(fz_gradient)
#     # print("If regrasp??: K2: ", np.sum(fz_gradient))
#     # ax2.set_yticks(np.arange(-0.5,0.5,0.1))
#     # ax2.axhline(y=-K1, color='gold', alpha=.7, linestyle='dashed')
#     # ax2.axhline(y=K1, color='gold', alpha=.7, linestyle='dashed')
#     plt.show()
#     return curr_jnt, ft_out

# regrasp
# F = np.loadtxt("/home/hlab/Desktop/20230324134301_lift.txt")
# only one
# F = np.loadtxt("/home/hlab/Desktop/20230324224751_lift.txt")
# nothing
# ft_out = np.loadtxt("/home/hlab/Desktop/exp/20230416201620_lift.txt")
# fs.plot(_path="/home/hlab/Desktop/exp/20230416213423_transport.txt", filter=True,animation=True)
fs.plot(_path="/home/hlab/bpbot/data/force/raw_20230425210300_0.txt", filter=True,animation=True)
