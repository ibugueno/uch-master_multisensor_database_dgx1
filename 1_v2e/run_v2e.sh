#!/bin/bash

# Archivo que contiene los directorios
INPUT_FILE="input/events_folders.txt"

# Activar el entorno conda
source /opt/conda/etc/profile.d/conda.sh
conda activate v2e

# Leer cada línea del archivo
while IFS= read -r line; do
    # Determinar height y width según el nombre del directorio
    if [[ "$line" == *"davis346"* ]]; then
        height=260
        width=346
    elif [[ "$line" == *"evk4"* ]]; then
        height=720
        width=1280
    else
        echo "No se reconoció el tipo de directorio para: $line"
        continue
    fi

    # Iterar sobre los modelos: clean y noisy
    #for model in clean noisy; do
    for model in clean; do
        # Construir los directorios de salida
        output_folder="output/$model/$line"

        # Ejecutar el comando
        python v2e.py -i "input/$line" \
            --overwrite \
            --input_frame_rate=1000 \
            --auto_timestamp_resolution=True \
            --dvs_exposure duration 0.005 \
            --output_folder="$output_folder" \
            --pos_thres=.15 \
            --neg_thres=.15 \
            --sigma_thres=0 \
            --output_width=$width \
            --output_height=$height \
            --disable_slomo \
            --no_preview \
            --skip_video_output \
            --dvs_aedat4 events.aedat4 \
            --dvs_params "$model"
    done

done < "$INPUT_FILE"
