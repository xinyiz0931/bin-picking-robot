import numpy as np
from bpbot.utils import *
root_dir = "/home/hlab/bpbot/data/calibration"
calib_path = "/home/hlab/bpbot/data/calibration/calibmat.txt"
from bpbot.config import BinConfig
bincfg = BinConfig()
cfg = bincfg.data
pixel_pose = [
[1145,748],
[ 958,751],
[ 964,544],
[ 789,736],
[1056,900],
[1304,516],
[1347,761],
[1348,941],
[1542,939]
]

# robot_pos_clb = [
#     [0.512, -0.011, 0.275],
#     [0.512, -0.111, 0.275],
#     [0.402, -0.101, 0.275],
#     [0.500, -0.201, 0.275],
#     [0.589, -0.061, 0.275],
#     [0.397,  0.073, 0.275],
#     [0.520,  0.089, 0.275],
#     [0.612,  0.089, 0.275],
#     [0.612,  0.189, 0.275]
# ]
robot_pos_clb = [
    [0.512, -0.011, 0.110],
    [0.512, -0.111, 0.110],
    [0.402, -0.101, 0.110],
    [0.500, -0.201, 0.110],
    [0.589, -0.061, 0.110],
    [0.397,  0.073, 0.110],
    [0.520,  0.089, 0.110],
    [0.612,  0.089, 0.110],
    [0.612,  0.189, 0.110]
]
import bpbot.driver.phoxi.phoxi_client as pclt
pxc = pclt.PhxClient(host="127.0.0.1:18300")
pxc.triggerframe()
pcd = pxc.getpcd()

camera_pos_clb = []
for p in pixel_pose:
    x, y = p
    camera_p = pcd[y*2064+x]/1000
    camera_pos_clb.append(camera_p)

camera_pos_clb = np.asarray(camera_pos_clb)
robot_pos_clb = np.asarray(robot_pos_clb)

print(camera_pos_clb.shape, robot_pos_clb.shape)
R, t = rigid_transform_3D(camera_pos_clb.T, robot_pos_clb.T)

print(camera_pos_clb.shape, robot_pos_clb.shape)
print("----------------------")
H = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
print(H)
print("----------------------")
np.savetxt(calib_path, H, fmt='%.06f')
# mat = np.loadtxt(calib_path)
# print(mat)
# # R = rpy2mat([0.12,0.82,0], seq='yxz')
# # R = rpy2mat([0,0,0], seq='yxz')
# R = rpy2mat([0,0,-2.5], seq='xyz')
# print(R)
# v = np.array([0.405,-0.154,0.030])

# _transform = np.r_[np.c_[R, v-np.dot(R,v)], [[0, 0, 0, 1]]]
# print("---------result-----------")
# mat = np.dot(mat, _transform)


# np.savetxt("/home/hlab/bpbot/data/calibration/calibmat.txt", mat, fmt='%.06f')
# print(mat)
