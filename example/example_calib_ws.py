"""
A Python scripts to set the workspace size 
Author: xinyi
Date: 20210721
Usage: `python utils/workspace_util.py vision/depth/depth.png`
---
Draw a rectangle for inside of the box. 
If you are happy with the size, hit `enter` or `q`, the config file will be updated. 
If you want re-draw the rectangle, hit `r` to refresh. 
"""
from email.policy import default
import sys
# execute the script from the root directory etc. ~/src/myrobot
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import argparse
from myrobot.config import BinConfig
from myrobot.utils import *

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        # cv2.imshow("image", image)

root_dir = os.path.abspath("./")
img_path = os.path.join(root_dir, "data/depth/depth_raw.png")

config_path = os.path.join(root_dir, "cfg/config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="add source image", default=img_path)
args = parser.parse_args()

img_path = os.path.join(root_dir, args.image)

ref_point = []
image = cv2.imread(img_path)
clone = image.copy()
cv2.namedWindow("window",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("window", shape_selection)
cv2.resizeWindow('window', 1920,1080)

while True:
    cv2.imshow("window",image)
    key=cv2.waitKey(1) & 0xFF

    # press r to reset the window
    if key == ord("r"):
        image = clone.copy()

    elif key == ord("q"):
        flag = 0
        break
    elif key==13: #enter
        flag = 1
        break

cv2.destroyAllWindows()

print('left_margin:   ', str(ref_point[0][0]))
print('top_margin:    ', str(ref_point[0][1]))
print('right_margin:  ', str(ref_point[1][0]))
print('bottom_margin: ', str(ref_point[1][1]))

# TODO: write to config file ...



# record the info into cfg file
# set absolute directory for config.yaml
if flag:
    cfg = BinConfig(config_path)
    cfg.set('left_margin', ref_point[0][0])
    cfg.set('top_margin', ref_point[0][1])
    cfg.set('right_margin', ref_point[1][0])
    cfg.set('bottom_margin', ref_point[1][1])
    cfg.write()
    main_proc_print("Successfully defined the workspace size! ")
else:
    warning_print("Failed to define the workspace size! ")
