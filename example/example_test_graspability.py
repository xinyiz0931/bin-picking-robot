import os
import sys
import math
import random
# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime as dt

from example.binpicking import *
from driver.phoxi import phoxi_client as pclt
from grasping.graspability import Gripper, Graspability
from motion.motion_generator import Motion
import learning.predictor.predict_client as pdclt
from utils.base_utils import *
from utils.transform_utils import *
from utils.vision_utils import *

def main():

    ROOT_DIR = os.path.abspath("./")
    img_path = os.path.join(ROOT_DIR, "vision/depth/depth0.png")

    gripper = Gripper()

    rotation_step = 45
    depth_step = 50
    hand_depth = 50

    margins = (0,0,500,500)
    g_params = (rotation_step, depth_step, hand_depth)
    grasps, input_img, full_image = detect_grasp_point(gripper=gripper, n_grasp=10, img_path=img_path, margins=margins, g_params=g_params)


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    