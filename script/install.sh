#!/bin/bash
# install pip for python2.7, for robot control
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
python get-pip.py
pip install -r ../requirements.txt
python -m pip install grpcio
python -m pip install protobuf
conda install -c conda-forge pybind11
#pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install tensorflow-gpu==2.2.0 tensorflow-probability==0.13.0
# pip install keras==2.3.0
