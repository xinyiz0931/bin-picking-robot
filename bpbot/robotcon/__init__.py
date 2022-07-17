import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'nxt'))
from .nxt.nxtrobot_client import NxtRobot
