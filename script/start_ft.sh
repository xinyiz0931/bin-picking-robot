#!/bin/bash

echo "[*] Start force sensor from bash"
export bashvar=100

cat << EOF > pyscript.py
#!/home/hlab/anaconda3/envs/xy/bin/python
from bpbot.device import FTSensor
print ('Monitoring from force/torque sensor...')
sensor = FTSensor()
sensor.record(plot=False)

EOF

chmod 755 pyscript.py

gnome-terminal -- bash -c "./pyscript.py"
