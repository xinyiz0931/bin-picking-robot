import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from . import asp_server
from . import asp_client
from .asp_server import ASPServer
from .asp_client import ASPClient