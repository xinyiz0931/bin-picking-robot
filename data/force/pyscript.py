#!/home/hlab/anaconda3/envs/xy/bin/python
import time
from bpbot.device import DynPickControl
print ('Record from force/torque sensor...')
sensor = DynPickControl()
sensor.record(plot=False)

