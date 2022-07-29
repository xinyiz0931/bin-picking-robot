import os
import cv2
import math
import random
import matplotlib.pyplot as plt
from bpbot import BinConfig
from bpbot.binpicking import *
from bpbot.utils import *
import open3d as o3d

from bpbot.grasping import Graspability, Gripper
import numpy as np
root_dir = os.path.abspath("./ext/bpbot")
root_dir = os.path.abspath("./")
img_path = os.path.join(root_dir, "data/depth/depth_cropped_mid_zone.png")
# img = cv2.imread(img_path)
# print((img > 35).sum())

# img_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")

bincfg = BinConfig(config_path)
cfg = bincfg.data
h_params = cfg["hand"]

calib_dir = os.path.join(root_dir, "data/calibration/try")

mat = np.loadtxt(os.path.join(calib_dir, "calibmat.txt"))
# # mat = np.loadtxt(os.path.join(root_dir, "data/calibration/calibmat.txt"))
print(mat)
rpoint = np.loadtxt(os.path.join(calib_dir, "robot.txt"))
rpoint[:,0] += 0.079
rpoint[:,2] -= 0.030
# # rpoint[:,2] += (0.014+0.017+0.135+0.0017)
cpoint = np.loadtxt(os.path.join(calib_dir, "camera.txt"))
# # rpoint = rpoint[0:10]
# # cpoint = cpoint[0:10]
# R, t = rigid_transform_3D((cpoint/1000).T, rpoint.T)
# mat = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
# print(mat)

# cpoint[:,2] += (0.017+0.014+0.135+0.0017)
fig = plt.figure()
print("Let's draw a cubic using o3d.geometry.LineSet.")
lines = []
for i in range(rpoint.shape[0]-1):
    lines.append([i,i+1])
_r = [[1, 0, 0] for i in range(len(lines))]
_b = [[0, 0, 1] for i in range(len(lines))]
# plot robot coordinate
rline_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(rpoint),
    lines=o3d.utility.Vector2iVector(lines),
)
rline_set.colors = o3d.utility.Vector3dVector(_r)
# plot camera coordinate
cline_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(cpoint),
    lines=o3d.utility.Vector2iVector(lines),
)
cline_set.colors = o3d.utility.Vector3dVector(_b)

# plot rotated camera coordinate 
# rot_cpoint = []
# for p in cpoint:
#     cx_, cy_, cz_,_ = np.dot(mat, [p[0],p[1],p[2],1])
#     rot_cpoint.append([cx_, cy_, cz_])
# rot_cline_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(rot_cpoint),
#     lines=o3d.utility.Vector2iVector(lines),
# ) 

rot_rpoint = []
for p in rpoint:
    rx_, ry_, rz_, _ = np.dot(np.linalg.inv(mat), [p[0], p[1], p[2], 1])
    rot_rpoint.append([rx_, ry_, rz_])
rot_rline_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(rot_rpoint),
    lines=o3d.utility.Vector2iVector(lines),
) 
rot_rline_set.colors = o3d.utility.Vector3dVector(_r)
# plot
pcd = o3d.io.read_point_cloud("/home/hlab/Desktop/test_ply.ply")
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]) 
# o3d.visualization.draw_geometries([mesh_frame, pcd, cline_set,rot_rline_set])
o3d.visualization.draw_geometries([mesh_frame, pcd])


# plot axis in robot coordinate

# o3d.visualization.draw_geometries([mesh_frame, rline_set, rot_cline_set])


# test plot


# img = cv2.imread(img_path)
# # img = adjust_grayscale(img)
# clone = img.copy()
# cropped_height, cropped_width, _ = img.shape

# finger_w = h_params['finger_width']
# finger_h = h_params['finger_height']
# open_w = h_params['open_width'] + random.randint(-1, 5)
# template_size = h_params['template_size']

# gripper = Gripper(finger_w=finger_w, 
#                     finger_h=finger_h, 
#                     open_w=open_w, 
#                     gripper_size=template_size)

# point = []
# def on_click(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(clone,(x,y),5,(0,255,0),-1)
#         print(f"{x},{y}")
#         point.append(x)
#         point.append(y)
# cv2.namedWindow("window")
# cv2.setMouseCallback("window", on_click)
# while(point == []):
#     cv2.imshow("window", clone)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27 or k==ord('q'):
#         break
# cv2.destroyAllWindows()
# angle_degree = gripper.point_oriented_grasp(img, [point[0], point[1]])

# grasp = [[point[0], point[1], angle_degree*math.pi/180]]
# drawn = draw_grasps(grasp, img_path, h_params, top_color=(0,255,0), top_only=True)
# plt.imshow(drawn), plt.show()
# # cv2.imwrite(draw_path, crop_grasp_pz)