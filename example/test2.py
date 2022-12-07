import numpy as np
from bpbot.config import BinConfig
from bpbot.binpicking import *
import pickle
with open('/home/hlab/Desktop/exp/20221201081040/out.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

print(unserialized_data)