# MOCAP

We have been using 2 different Deep Learning Networks in this Motion Capture repo.

1. [3D Pose GAN](https://github.com/DwangoMediaVillage/3dpose_gan)
2. [End-to-end Recovery of Human Shape and Pose](https://github.com/akanazawa/hmr)

For the 2nd, we are specifically using this [fork](https://github.com/Dene33/hmr)

## Installation

```
git clone https://github.com/abdullahchand/MOCAP.git
cd MOCAP
./download_models.sh
```

## Setup Environments

Both Deep Networks requires different environments, 3D Pose GAN is python 3.x based while HMR is 2.7 so different Conda Environments for both have been setup, requirments for each could be installed as:

```
conda create -n 3dposegan python==3.5
conda install --file requirements_3dposegan.txt

conda create -n HMR python==2.7
conda install --file requirements_hmr.txt
```

## Config

The pipeline executes with 3dposegan environment but to switch from HMR env, the python script ```run_hmr_demo.py``` needs to be configure as:

```subprocess.run(['path/to/conda/python2.7', 'hmrdemo.py'])```
For Example: ```/home/ubuntu/anaconda3/envs/py27/bin/python```

## Run

Place your input video in ```video/``` folder with name ```input.mp4``` and execute:

```
./movrs.sh
```
To run API on localhost:
```
python flask_main.py
```
