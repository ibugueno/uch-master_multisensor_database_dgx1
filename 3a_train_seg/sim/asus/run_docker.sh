#!/bin/bash

docker rm -fv ignacio_3a_train_seg_asus_sim

docker run -it --shm-size=16G --gpus '"device=5"' --name ignacio_3a_train_seg_asus_sim -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s/models:/app/output ignacio_3a_train_seg_asus_sim