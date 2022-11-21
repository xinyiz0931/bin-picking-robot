import math
import numpy as np
from bpbot.utils import sample_uniformly

J = np.loadtxt("/home/hlab/bpbot/data/motion/motion_ik.dat")
# H = sample_uniformly(J, itvl=2)

LJ = J[3:6][:,10:16]
RJ = J[3:6][:,4:10]
RJ = RJ  / 180 * math.pi
print(RJ)
# J = np.array(
#     [[-0.211185, -0.656244, -1.532399, -0.380482, 0.479966, 0.26529],
#      [-0.282743, -0.858702, -1.53938, -0.492183, 0.521853, 0.190241],
#      [-0.397935, -1.001819, -1.665044, -0.680678, 0.610865, 0.071558]]
# )


# J = np.array([[1,2,3,4,5], [2,4,7,2,4]])
H = RJ
print(len(H), "motions! ")
with open("/home/hlab/Desktop/lhand.txt", 'w') as fp:
    print("robot.playPatternOfGroup('larm',", file=fp)
    print('[', end='', file=fp)
    for i, h in enumerate(H):
        if i < len(H) - 1:
            print('['+','.join(str(i) for i in h)+'],', file=fp)
        else:
            print('['+','.join(str(i) for i in h)+']],', file=fp)
    print("[3"+",1"*len(H)+"])", file=fp)
# np.savetxt("/home/hlab/Desktop/lhand.txt", H, fmt='%.06f', delimiter=',')
# itvl = 5
# newJ = []
# for i, jnt in enumerate(J):
#     if i == len(J) - 1:
#         newJ.append(jnt)
#         break
#     newJ.append(jnt)
#     jnt_next = J[i+1] 
#     _delta = (jnt_next-jnt)/itvl
#     for k in range(itvl-1):
#         newJ.append(_delta*(k+1)+jnt)
# newJ = np.array(newJ)
# print(J.shape, " => ", newJ.shape)
# print(newJ)