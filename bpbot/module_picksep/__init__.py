import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from . import picksep_client
from .picksep_client import PickSepClient