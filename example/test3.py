import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from tangle.dataset import SepDataset
from tangle.utils import *
from tangle import Config


cfg = Config(config_type="train")
# data_folder = cfg.data_dir

# data_inds = random_inds(10,1000)
data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEightAugment"

_search = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel\\U\\**\\depth.png"
_dest = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel\\_vis"
import glob
import shutil
num_f = 0
num_p = 0
for d in glob.glob(_search, recursive=True):
    j_path = os.path.join(*os.path.split(d)[:-1], "sln.json")
    print(os.path.exists(j_path))
    if os.path.exists(j_path): 
        fp = open(j_path, 'r+')
        j_file = json.loads(fp.read())
        num_f += 1
        if "pull" in j_file and "hold" in j_file:
            
            new_d = os.path.join(_dest, d.split('\\')[-2]+'.png')
            num_p += 1 
            print(new_d, j_file["pull"], j_file["hold"])
            # img = cv2.imread(d)
            # cv2.circle(img, j_file["pull"], 7, (0,255,0), -1)
            # cv2.circle(img, j_file["hold"], 7, (0,255,255), 2)
            # cv2.imwrite(new_d, img)
            
print("exist: ", num_f)
print("point: ", num_p)
            # print(d, '->', new_d)
        #     shutil.copyfile(d, new_d)