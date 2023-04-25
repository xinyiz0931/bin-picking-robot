import os
import random
import importlib
spec = importlib.util.find_spec("cnoid")
FOUND_CNOID = spec is not None
if FOUND_CNOID: 
    from cnoid.Util import *
    from cnoid.Base import *
    from cnoid.Body import *
    from cnoid.BodyPlugin import *
    from cnoid.GraspPlugin import *
    from cnoid.BinPicking import *
    topdir = executableTopDirectory
else: 
    topdir = "/home/hlab/choreonoid-1.7.0/"

from bpbot.binpicking import *
from bpbot.config import BinConfig
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot
from bpbot.utils import * 
import timeit
import numpy as np
start = timeit.default_timer()

cfg = BinConfig()
cfgdata = cfg.data

G = np.loadtxt(cfgdata["calibmat_path"])

point_array = capture_pc()
img = transform_depth(point_array, G, 0.5, 0.45, cfgdata["height"], cfgdata["width"])
h_ = [394, 1203]
w_ = [761, 1511]
crop = img[394:1203, 761:1511]
crop = img[h_[0]:h_[1], w_[0]:w_[1]]
ratio = np.count_nonzero(crop) / ((h_[1]-h_[0])*(w_[1]-w_[0]))
print("Nonzero: ", ratio)
cv2.imshow("window", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()