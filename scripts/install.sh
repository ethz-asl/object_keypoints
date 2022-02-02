#!/bin/bash

[ -n "$OBJECT_KEYPOINTS_DIR" ] || fail "Please set the OBJECT_KEYPOINTS_DIR variable first"

cd

if [ ! -d "miniconda3" ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  rm Miniconda3-latest-Linux-x86_64.sh
fi

source miniconda3/bin/activate
conda create -n env Python=3.8 numpy
source miniconda3/bin/activate
conda activate env

conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch
conda install ffmpeg -y

cd $OBJECT_KEYPOINTS_DIR

pip install -r requirements.txt
pip install -e .

cd perception/corner_net_lite/core/models/py_utils/_cpools/
pip install .
#python3 ./setup.py install --user

cd

echo "Please run 'source miniconda3/bin/activate' and 'conda activate env' every time you want to use the package"

