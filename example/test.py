import os
import cv2
import math
import random
import matplotlib.pyplot as plt
from bpbot import BinConfig
from bpbot.binpicking import *



from bpbot.grasping import Graspability, Gripper
import numpy as np
root_dir = os.path.abspath("./ext/bpbot")
img_path = os.path.join(root_dir, "data/depth/depth_cropped_mid_zone.png")
# img = cv2.imread(img_path)
# print((img > 35).sum())

# img_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")

bincfg = BinConfig(config_path)
cfg = bincfg.config
h_params = cfg["hand"]

# test plot
points = np.loadtxt("/home/hlab/Desktop/finger.txt")
print(points.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.scatter(points[:,0], points[:,1], points[:,2], color='red')
s= 0.2
ax.set_xlim3d(-s,s)
ax.set_ylim3d(-s,s)
ax.set_zlim3d(-s,s)
plt.show()

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