import numpy as np
import os
import cv2

root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))

# TODO visualization 
# for pick
SIZE = 1000
HALF_SIZE = int(SIZE/2)
h = cv2.imread("/home/hlab/bpbot/data/depth/depth.png")
from bpbot.utils import cv_plot_title
cv_plot_title(h, "Affordance Map")