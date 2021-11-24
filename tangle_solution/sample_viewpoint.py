import os
import sys
sys.path.append("./")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from utils.plot_utils import WireframeSphere

if __name__ == '__main__':
    for raw, yall in itertools.product([-90,0,90], repeat=2):
        print([raw, 0, yall])