#!/bin/bash

docker rm -fv ignacio_0.5_others_scenarios_2

docker run -it --name ignacio_0.5_others_scenarios_2 -v /home/ignacio.bugueno/cachefs/tesis/blender:/app/input -v /home/ignacio.bugueno/cachefs/tesis/data/ddbb-s:/app/output ignacio_0.5_others_scenarios_2
