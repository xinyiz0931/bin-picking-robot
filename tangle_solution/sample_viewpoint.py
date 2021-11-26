import os
import sys
sys.path.append("./")
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from utils.plot_utils import WireframeSphere

if __name__ == '__main__':
    
    ano = cv2.imread("D:\\code\\myrobot\\vision\\depth\\depth0.png", 0)
    black = np.zeros(ano.shape)
    plt.imshow(ano, cmap='gray')
    plt.imshow(black, cmap='jet', alpha = 0.4)
    plt.axis('off')
    plt.show()
