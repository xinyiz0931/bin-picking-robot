#!/bin/bash

echo "[*] Stop force sensor from bash"
gnome-terminal -- bash -c "killall -9 pyscript.py"
