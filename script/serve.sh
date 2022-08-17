#!/bin/bash
gnome-terminal  --tab -e "bash -c 'python ./bpbot/driver/phoxi/phoxi_server.py; bash'" --tab -e "bash -c 'python ./bpbot/module_picksep/picksep_server.py; bash'"
