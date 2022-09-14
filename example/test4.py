import numpy as np
import matplotlib.pyplot as plt
h_ = np.array([
          [ 0.537,  0.160, 0.320], 
          [ 0.060, -0.100, 0.050],  
          [ 0.060, -0.100, 0.050], 
          [-0.100, -0.150, 0.000], 
          [-0.020,  0.350, 0.000]
     ])

# hh = np.array([
#      [0.537,  0.160, 0.320],
#      [0.597,  0.060, 0.350],
#      [0.497, -0.090, 0.370],
#      # [0.477,  0.240, 0.370]
# ])

hh = np.array([
     [0.525,  0.290, 0.320],
     [0.620,  0.160, 0.370],
     [0.626, -0.040, 0.400],
     [0.531, -0.160, 0.420],
     # [0.537,  0.160, 0.420]
])
h = np.array([
     [0.525,  0.290, 0.320], # o
     [0.611,  0.177, 0.340], # half
     [0.626, -0.048, 0.360], # full
     [0.525, -0.160, 0.380], # full
     [0.431, -0.051, 0.380], # full
     [0.423,  0.184, 0.400], # half
     [0.525,  0.290, 0.410], # o
     # [0.618,  0.179, 0.420], # half
     # [0.624, -0.045, 0.430], # full
     # [0.530, -0.162, 0.440], # full
     # [0.432, -0.050, 0.450], # full
     # [0.419,  0.183, 0.460], # half
     # [0.525,  0.288, 0.460], # o

])

def plot(hh, ax):

     ax.scatter3D(hh[:,0], hh[:,1], hh[:,2], color='r', alpha=1)
     ax.plot(hh[:,0], hh[:,1], hh[:,2], color='r', alpha=1)
     
     ax.scatter3D(hh[:,0], hh[:,1], np.zeros(len(hh)), color='k', alpha=0.5)
     ax.plot(hh[:,0], hh[:,1], np.zeros(len(hh)), color='k', alpha=0.5)

     for p in hh:
          ax.plot([p[0],p[0]],[p[1],p[1]],[0,p[2]], color="k", alpha=0.3, linestyle='dashed')
     
     ax.view_init(24, 26)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# plot table / plane
X, Y = np.meshgrid(np.arange(0.4,0.7,0.05), np.arange(-0.20, 0.35,0.05))
Z = 0*X
ax.plot_surface(X, Y, Z, color='grey', alpha=0.3)  # the horizontal plane
# plot ideal trajectory
from matplotlib.patches import Circle, Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
# p = Circle((0.525, 0.065), 0.1)
p = Ellipse((0.525, 0.065), 0.1*2, 0.225*2, fill=False, color='r', linestyle='--', linewidth=1.5)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
# plot center
c = [0.525, 0.065]
ax.scatter3D(c[0], c[1], 0, color='r', alpha=0.5)
plot(hh, ax)

# plot axis
# ax.quiver(0, 0, 0, 0.05, 0, 0, color='r', arrow_length_ratio=0.05, alpha=0.3) # x-axis
# ax.quiver(0, 0, 0, 0, 0.05, 0, color='g', arrow_length_ratio=0.05, alpha=0.3) # y-axis
# ax.quiver(0, 0, 0, 0, 0, 0.05, color='b', arrow_length_ratio=0.05, alpha=0.3) # z-axis

# final reformation
plt.xlim([0.3,0.65])
plt.xticks(np.arange(0.3, 0.7, 0.1))
plt.yticks(np.arange(-0.2, 0.3, 0.1))
plt.ylim([-0.2, 0.3])
labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels] # 设置刻度值字体

ax.tick_params(axis='both', which='major', labelsize=25)
plt.rcParams.update({'font.family':'Times New Roman'})
plt.show()
