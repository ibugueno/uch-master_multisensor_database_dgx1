#!/bin/bash

docker rm -fv ignacio_3b_train_det_asus_sim

docker run -it --gpus '"device=3"' --name ignacio_3b_train_det_asus_sim -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s/models:/app/output ignacio_3b_train_det_asus_sim
