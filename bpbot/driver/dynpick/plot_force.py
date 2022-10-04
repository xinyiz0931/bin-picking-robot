import os
import time
import glob
import matplotlib.pyplot as plt
import numpy as np

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
# start recording
def plot_animation(force, x):

    f_new = np.asarray(force)
    ax1 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_prop_cycle(color=colors)
    #major_ticks = np.arange(x[0], x[-1], 10 if len(x) < 160 else 20)
    #minor_ticks = np.arange(x[0], x[-1], 1 if len(x) < 160 else 2)
    major_ticks = np.arange(x[0], x[-1], 10)
    
    #minor_ticks = np.arange(x[0], x[-1])
    hline_c = 'gold'

    #for i, f in enumerate(f_new):
    #    if i==0: continue
    #    if f[2] > f_new[i-1][2] and f[2] > 0.1:
    #        ax1.axvline(x=x[i], color=hline_c, alpha=1)
    #        ax1.axhline(y=0, color=colors[0], alpha=.5, linestyle='dashed')
    #        #print(i)
        #if abs(f[1]) > 4.8:
        #    ax1.axvline(x=x[i], color=hline_c, alpha=1)
        #    ax1.axhline(y=4.8 if f[1] > 0 else -4.8, color=colors[1], alpha=.5, linestyle='dashed')
        #    print(i)
        #if abs(f[2]) > 6:
        #    ax1.axvline(x=x[i], color=hline_c, alpha=1)
        #    ax1.axhline(y=6 if f[2] > 0 else -6, color=colors[2], alpha=.5, linestyle='dashed')
        #    print(i)
    #ax1.axhline(y=4, color=colors[0], alpha=.5, linestyle='dashed')
    ax1.set_title('Force')
    ax1.set_xticks(major_ticks)

    #ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    ax1.legend()
    #handles, labels = ax1.get_legend_handles_labels()
    #ax1.legend(handles=handles, labels=eval(labels[0]), loc='upper left')
    plt.ion()
    #plt.ylim(-2, 2)
    plt.ylim(-0.5, 0.5)
    plt.show()
fig = plt.figure(1)

ax1 = fig.add_subplot(311)
init = [7984,8292,8572]
plot_data = []
j = 0
# 1
F = np.loadtxt("./out_no_picking.txt")
print(F.shape)
F /= 1000
F = F[71:]
X = F[:,0] - F[0][0]
ax1.plot(X, F[:,1]-init[0]/1000, color=colors[1], label="Fx")
ax1.plot(X, F[:,2]-init[1]/1000, color=colors[2], label="Fy")
ax1.plot(X, F[:,3]-init[2]/1000, color=colors[0], label="Fz")
ax1.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')
plt.ylim(-0.5,0.5)
plt.xlim(0, 12)
plt.legend(loc='upper right')
ax1.set_xticks([])

#2
ax2= fig.add_subplot(312)
F = np.loadtxt("./out_nobutyes_picking.txt")
print(F.shape)
F /= 1000
F = F[51:]
X = F[:,0] - F[0][0]
ax2.plot(X, F[:,1]-init[0]/1000, color=colors[1])
ax2.plot(X, F[:,2]-init[1]/1000, color=colors[2])
ax2.plot(X, F[:,3]-init[2]/1000, color=colors[0])
ax2.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')
ax2.set_xticks([])
plt.ylim(-0.5, 0.5)
plt.xlim(0, 12)

#3
ax3= fig.add_subplot(313)
F = np.loadtxt("./out_yes_picking.txt")
print(F.shape)
F /= 1000
F = F[34:]
X = F[:,0] - F[0][0]
ax3.plot(X, F[:,1]-init[0]/1000, color=colors[1])
ax3.plot(X, F[:,2]-init[1]/1000, color=colors[2])
ax3.plot(X, F[:,3]-init[2]/1000, color=colors[0])
ax3.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')
plt.ylim(-0.5, 0.5)
plt.xlim(0, 12)

plt.rcParams.update({'font.family':'Times New Roman'})
plt.tight_layout()
plt.show()

# for data in F:
#     plot_data.append([(data[1]-init[0])/1000, (data[2]-init[1])/1000, (data[3]-init[2])/1000])
#     j+=1 
#     fig = plt.figure(1, figsize=(16, 6))
#     if len(plot_data) <= 50:
#         force = [[0,0,0] for _ in range(50-j)] + plot_data
#     else:
#         force = plot_data[j-50:]
#     plt.clf()
#     force_tmp = np.asarray(force)
#     plot_animation(force_tmp, range(j,j+50))
#     plt.pause(0.005)
#     #time.sleep(.5)
