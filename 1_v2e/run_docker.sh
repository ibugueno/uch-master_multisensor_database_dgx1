#!/bin/bash

docker rm -fv ignacio_1_v2e

docker run -it --gpus '"device=0"' --name ignacio_1_v2e -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/blender_raw_events:/app/output ignacio_1_v2e
