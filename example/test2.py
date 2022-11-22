import numpy as np
import os
import cv2

root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))

# TODO visualization 
# for pick
SIZE = 1000
HALF_SIZE = int(SIZE/2)
h = cv2.imread(os.path.join(root_dir, "data/depth/pred/picknet_depth_cropped_pickbin.png"))
# heatmap = cv2.resize(h, (SIZE, int(h.shape[0]*SIZE/h.shape[1])))
upper = cv2.resize(h, (SIZE, HALF_SIZE))
black = np.zeros((HALF_SIZE, HALF_SIZE,3)).astype(np.uint8)
ret = cv2.imread(os.path.join(root_dir, "data/depth/result.png"))
ret = cv2.resize(ret, (HALF_SIZE, HALF_SIZE))
lower = cv2.hconcat([black, ret])
title = np.zeros((50,HALF_SIZE,3)).astype(np.uint8)
vis = cv2.vconcat([upper, lower])
import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(16, 6))
ax1 = fig.add_subplot(121)
ax1.imshow(h)
ax2 = fig.add_subplot(122)
ax2.imshow(ret)
plt.show()
# ret_pickbin = v2.imread(os.path.join(root_dir, "data/depth/pred/ret_depth_cropped_pick_zone.png"))
# vis = []
# for v in [heatmap_pickbin, cv2.hconcat([grasp_pickbin, img_grasp])]:
#     vis.append(cv2.resize(v, (1000, int(v.shape[0]*1000/v.shape[1]))))
# vis_pickbin = cv2.vconcat(vis)
# vis_dropbin = (np.ones([*vis_pickbin.shape])*255).astype(np.uint8)
# cv2.putText(vis_dropbin, "Bin (Drop)",(20,550), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
# cv2.putText(vis_dropbin, "No Action",(20,700), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
# vis = cv2.hconcat([vis_pickbin, vis_dropbin])
# cv2.imwrite(os.path.join(root_dir, "data/depth/vis.png"), vis)