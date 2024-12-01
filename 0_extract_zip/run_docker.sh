#!/bin/bash

docker rm -fv ignacio_0_extract_zip

nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0  --name ignacio_0_extract_zip -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/tmp ignacio_0_extract_zip

