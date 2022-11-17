import numpy as np
from bpbot.utils import sample_uniformly

# J = np.loadtxt("/home/hlab/bpbot/data/motion/motion_ik.dat")
J = np.array([[1,2,3,4,5], [2,4,7,2,4]])
print(sample_uniformly(J, itvl=10))
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