#!/bin/bash

docker rm -fv ignacio_1_v2e_2

docker run -it --gpus '"device=1"' --name ignacio_1_v2e_2 -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/blender_raw_events:/app/output ignacio_1_v2e_2
