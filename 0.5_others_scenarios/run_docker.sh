#!/bin/bash

docker rm -fv ignacio_0.5_others_scenarios

docker run -it --gpus '"device=1"' --name ignacio_0.5_others_scenarios -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s/2_without_back_blur:/app/output ignacio_0.5_others_scenarios
