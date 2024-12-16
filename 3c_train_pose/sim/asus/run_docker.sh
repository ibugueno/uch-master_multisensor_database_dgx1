#!/bin/bash

docker rm -fv ignacio_3c_train_pose_asus_sim

docker run -it --shm-size=16G --gpus '"device=0"' --name ignacio_3c_train_pose_asus_sim -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s/models:/app/output ignacio_3c_train_pose_asus_sim
