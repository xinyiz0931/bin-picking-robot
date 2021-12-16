import os
import sys


sys.path.append("./")
from example.binpicking import *

from grasping.graspability import Gripper
def main():

    ROOT_DIR = os.path.abspath("./")
    img_path = os.path.join(ROOT_DIR, "vision/depth/depth0.png")

    gripper = Gripper()

    rotation_step = 22.5
    depth_step = 50
    hand_depth = 50
    main_proc_print("Rotation step: {}".format(rotation_step))
    main_proc_print("Depth step: {}".format(depth_step))

    margins = (0,0,500,500)
    g_params = (rotation_step, depth_step, hand_depth)
    grasps, input_img, full_image = detect_grasp_point(gripper=gripper, n_grasp=10, img_path=img_path, margins=margins, g_params=g_params)

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    