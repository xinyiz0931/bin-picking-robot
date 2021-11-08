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
import sys
# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import argparse
import configparser

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



parser = argparse.ArgumentParser()
parser.add_argument("image", help="add source image")
args = parser.parse_args()

ROOT_DIR = os.path.abspath("./")
img_path = os.path.join(ROOT_DIR, args.image)
config_path = os.path.join(ROOT_DIR, "cfg/config.ini")

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
        break
    elif key==13: #enter
        break

cv2.destroyAllWindows()

# record the info into cfg file
# set absolute directory for config.yaml

config = configparser.ConfigParser()
config.read(config_path)

config.set('GRASP', 'left_margin', str(ref_point[0][0]))
config.set('GRASP', 'top_margin', str(ref_point[0][1]))
config.set('GRASP', 'right_margin', str(ref_point[1][0]))
config.set('GRASP', 'bottom_margin', str(ref_point[1][1]))

config.write(open(config_path, "w"))
print("Successfully defined the workspace size! ")