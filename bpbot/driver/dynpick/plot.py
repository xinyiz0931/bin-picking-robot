import time
import numpy as np
import matplotlib.pyplot as plt
def plot_f(force, path, x, objrelrot=None):
    colors = [[ 78.0/255.0,121.0/255.0,167.0/255.0], # 0_blue
              [255.0/255.0, 87.0/255.0, 89.0/255.0], # 1_red
              [ 89.0/255.0,169.0/255.0, 79.0/255.0], # 2_green
              [237.0/255.0,201.0/255.0, 72.0/255.0], # 3_yellow
              [242.0/255.0,142.0/255.0, 43.0/255.0], # 4_orange
              [176.0/255.0,122.0/255.0,161.0/255.0], # 5_purple
              [255.0/255.0,157.0/255.0,167.0/255.0], # 6_pink 
              [118.0/255.0,183.0/255.0,178.0/255.0], # 7_cyan
              [156.0/255.0,117.0/255.0, 95.0/255.0], # 8_brown
              [186.0/255.0,176.0/255.0,172.0/255.0]] # 9_gray

    f_new = []
    if objrelrot is None:
        f_new = force[:, :3]

    f_new = np.asarray(f_new)
    ax1 = fig.add_subplot(111)
    plt.ylim(-6, 8)
    # ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_prop_cycle(color=colors)
    major_ticks = np.arange(x[0], x[-1], 10 if len(x) < 160 else 20)
    minor_ticks = np.arange(x[0], x[-1], 1 if len(x) < 160 else 2)
    hline_c = 'gold'

    for i, f in enumerate(f_new):
        if f[0] < 0:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=0, color=colors[0], alpha=.5, linestyle='dashed')
            print(i)
        if abs(f[1]) > 4.8:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=4.8 if f[1] > 0 else -4.8, color=colors[1], alpha=.5, linestyle='dashed')
            print(i)
        if abs(f[2]) > 6:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=6 if f[2] > 0 else -6, color=colors[2], alpha=.5, linestyle='dashed')
            print(i)
    ax1.axhline(y=4, color=colors[0], alpha=.5, linestyle='dashed')
    ax1.set_title('Force')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    handles, labels = ax1.get_legend_handles_labels()
    #ax1.legend(handles=handles, labels=eval(labels[0]), loc='upper left')
    plt.ion()
    plt.show()

fig = plt.figure(1, figsize=(16, 6))

#force = info[0]
#path = info[1]
import random
#force = [[0, 0, 0, 0, 0] for _ in range(100)] + force
force = [[0,0,0,0,0] for _ in range(50)]+[[random.random()*2 for _ in range(5)] for _ in range(100)] 
path = [None for _ in range(100)]
for i in range(100):
    plt.clf()
    force_tmp = np.asarray(force[i:i + 50])
    path_tmp = np.asarray(path[i:i + 50])
    plot_f(force_tmp, path_tmp, range(i, i + 50))
    plt.pause(0.005)
    time.sleep(.1)
