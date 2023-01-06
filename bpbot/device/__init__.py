import os
import sys
import warnings
warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'phoxi'))
from .phoxi.phoxi_client import PhxClient
from .dynpick.ft_sensor import FTSensor
# from .phoxi import phoxi_server
# from .phoxi import phoxi_client
# from .phoxi.phoxi_server import PhoxiServer
# from .phoxi.phoxi_client import PhxClient
