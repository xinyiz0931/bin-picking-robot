#!/bin/bash
gnome-terminal  --tab -e "bash -c 'python ./bpbot/driver/phoxi/phoxi_server.py; bash'" --tab -e "bash -c '/usr/bin/python ./bpbot/robotcon/nxt/nxtrobot_server.py; bash'" --tab -e "bash -c 'python ./bpbot/prediction/predict_server.py; bash'"
