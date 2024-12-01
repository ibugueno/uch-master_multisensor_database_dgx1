#!/bin/bash

docker rm -fv ignacio_ec_v2e

nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0  --name ignacio_ec_v2e -v /home/ignacio.bugueno/cachefs/db/emulator/v2e/input:/app/input -v /home/ignacio.bugueno/cachefs/db/emulator/v2e/output:/app/output -v /home/ignacio.bugueno/cachefs/db/emulator/v2e/tmp:/app/tmp ignacio_ec_v2e

