#!/home/hlab/anaconda3/envs/xy/bin/python
from bpbot.device import FTSensor
print ('Monitoring from force/torque sensor...')
sensor = FTSensor()
sensor.record(plot=True)

