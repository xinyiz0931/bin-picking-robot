import os
import glob
import cv2
from bpbot.device import FTSensor
import matplotlib.pyplot as plt
import numpy as np
sensor = FTSensor()

root_dir = "/home/hlab/Desktop/newsend/2022*"
for d in glob.glob(root_dir):
    out_path = os.path.join(d, "force.txt")
    img_path = os.path.join(d, "grasp.png")
    img = cv2.imread(img_path)
    sensor.plot_file(_path=out_path)
    # plt.imshow(img)
    plt.show()
# sensor.plot_file(_path="/home/hlab/Desktop/newsend/20221222224201 vertical shaking works 2 same video second haf/force.txt")

# _data = np.loadtxt(sensor.out_path)
# sep_idx = np.where(np.any(_data==sensor.sep_kw, axis=1))[0]
# data = np.delete(_data, sep_idx, 0)
# sep_idx -= 1
# sep_x = _data[sep_idx][:,0]/1000
# for i in range(len(sep_idx)-1):
#     if i == 3 or i == len(sep_idx) -6  or i == len(sep_idx) -5:
#         d = data[sep_idx[i]:sep_idx[i+1]]
#         tm = d[:,0]/1000
#         ft = (d[:,1:]-sensor.zero)/1000
#         x = tm - tm[0]
#         sensor.plot_file(_data=d)


