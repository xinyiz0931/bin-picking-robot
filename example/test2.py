import os
import cv2
from bpbot.config import BinConfig
from bpbot.grasping import Gripper
from bpbot.binpicking import *
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

root_dir = "/home/hlab/bpbot" 
bincfg = BinConfig()
cfg = bincfg.data

schunk_attr = [cfg["hand"]["schunk"].get(k) for k in ["finger_width", "finger_height", "open_width"]]
gripper_schunk = Gripper(*schunk_attr)

img_path = os.path.join(root_dir, "data/depth/depth_cropped_drop_zone.png")
table_path = os.path.join(root_dir, "data/depth/depth_drop_zone.png")
img = cv2.imread(img_path)
table = cv2.imread(table_path)
pcd = o3d.io.read_point_cloud("/home/hlab/choreonoid-1.7.0/ext/graspPlugin/PCL/test_ply.ply") 
point_array = np.asarray(pcd.points)

m = cfg["drop"]["margin"]
mp = [[m["left"], m["top"]], [m["right"], m["top"]], 
                    [m["right"], m["bottom"]], [m["left"], m["bottom"]]]

p_r_pull = np.array([0.420, 0.087])
p_r_hold = np.array([0.446, 0.049])
p_pull = np.array([192,198])
p_hold = np.array([111,252])

mr = []
for p_i in mp:
    p_r = transform_image_to_robot(p_i, point_array, cfg)
    mr.append(p_r)
_l = 0.1
print(mr)
mr.append(mr[0])

mr = np.asarray(mr)
plt.plot(mr[:,0], mr[:,1], 'blue')

plt.scatter(*p_r_hold, color='gold')
plt.scatter(*p_r_pull, color='green')
V = np.array([-0.38, 0.92])
plt.plot([p_r_pull[0], p_r_pull*(1+_l)[0]],[p_r_pull[1], p_r_pull*(1+_l)[1]])
plt.show()

# p_pick = [344,231]

# g_pick = gripper_schunk.point_oriented_grasp(img, p_pick) # degree
# print("[*] Best grasp: ", g_pick)
# if g_pick is not None: 
#     img = draw_grasps(g_pick, img)
#     plt.imshow(img, cmap='gray')
#     plt.show()
# else:
#     print("No grasps! ")