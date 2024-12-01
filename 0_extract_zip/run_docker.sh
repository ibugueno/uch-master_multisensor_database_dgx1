#!/bin/bash

docker rm -fv ignacio_0_extract_zip

docker run -it --gpus '"device=0"' --name ignacio_0_extract_zip -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/tmp ignacio_0_extract_zip
