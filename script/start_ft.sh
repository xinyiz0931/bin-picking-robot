#!/bin/bash

echo "[*] Start force sensor from bash"
export bashvar=100

cat << EOF > pyscript.py
#!/home/hlab/anaconda3/envs/xy/bin/python
import time
from bpbot.device import DynPickControl
print ('Record from force/torque sensor...')
sensor = DynPickControl()
sensor.record(plot=False)

EOF

chmod 755 pyscript.py

gnome-terminal -- bash -c "./pyscript.py"
