import os
import sys
sys.path.append("./")
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

from utils.transform_utils import *

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

if __name__ == '__main__':
    phi=45
    theta = 45
    # start = -90
    # end=90 + 0.1*phi




    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    views = []
    # for x in np.arange(start, end, phi):
    #     r = R.from_euler('x', x, degrees=True)
    #     angle = r.as_euler('xyz', degrees=True)
    #     if (angle != [0,0,0]).any():
    #         views.append(angle)
    # for y in np.arange(start, end, phi):
    #     r = R.from_euler('y', y, degrees=True)
    #     angle = r.as_euler('xyz', degrees=True)
    #     if (angle != [0,0,0]).any() :
    #         views.append(angle)
    # for z in np.arange(start, end, phi):
    #     r = R.from_euler('z', z, degrees=True)
    #     angle = r.as_euler('xyz', degrees=True)
    #     if (angle != [0,0,0]).any() :
    #         views.append(angle)
    
    init_view = [0,0,1]
    ax.quiver(0, 0, 0, init_view[0], init_view[1], init_view[2], length = 2, color='black', alpha=0.75)
    for x in np.arange(0,91,phi):
        for z in np.arange(0,360,theta):
            if x != 0:
                # print(x,0,z) # euler angle
                rot = np.dot(rpy2mat([x,0,z]), init_view)
                rot =  rot / np.linalg.norm(rot)
                ax.quiver(0, 0, 0, rot[0], rot[1], rot[2], length = 2, color='black', alpha=0.25,lw=2)

        # for v in views:
        #     if (q!=v).any():
        #         views.append(q)

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 2)
    ax.set_box_aspect(aspect = (2,2,1))     
    # plt.axis('off')
    ax.plot([0, 0.5], [0, 0], [0, 0], color='red') # x
    ax.plot([0, 0], [0, 0.5], [0, 0], color='green') # y
    ax.plot([0, 0], [0, 0], [0, 0.5], color='blue') # z
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(27,28)
    plt.show()
