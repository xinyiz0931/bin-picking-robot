import cv2
import bpbot.driver.phoxi.phoxi_client as pclt
from bpbot.utils import *
pxc = pclt.PhxClient(host="127.0.0.1:18300")
pxc.triggerframe()
gray = pxc.getgrayscaleimg()
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

pcd = pxc.getpcd()
id_locs = detect_ar_marker(image.copy(), "DICT_5X5_100", show=True)
# top, bottom, left, right
ids = [35, 43, 48, 33]

# top
x, y = id_locs[35]
p_top = pcd[y*image.shape[1]+x]
# bottom
x, y = id_locs[43]
p_bottom = pcd[y*image.shape[1]+x]
# left
x, y = id_locs[48]
p_left = pcd[y*image.shape[1]+x]
# rihgt
x, y = id_locs[33]
p_right = pcd[y*image.shape[1]+x]

# calculate 5 other points
p_c = (p_top + p_bottom + p_left + p_right)/4
x_, y_ = 1177,724
print(f"center: {p_c}, manual: {pcd[y_*image.shape[1]+x_]}")
