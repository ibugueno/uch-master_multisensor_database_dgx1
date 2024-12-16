#!/bin/bash

# Nombre base de las sesiones de tmux y el script a ejecutar
SESSION_BASE="extract_images"
SCRIPT="4_extract_images.py"

SENSOR="davis346"
MODEL="clean"

# Ruta al entorno virtual base (ajusta según tu configuración)
CONDA_ACTIVATE="source ~/anaconda3/bin/activate base"

# Crear sesiones de tmux para cada clase
for i in {0..19}; do
    SESSION="${SESSION_BASE}_${i}"
    OBJECT_CLASS=$i

    # Crear y configurar la sesión de tmux
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION "$CONDA_ACTIVATE" C-m  # Activa el entorno virtual base
    tmux send-keys -t $SESSION "python $SCRIPT $OBJECT_CLASS $MODEL $SENSOR" C-m  # Ejecuta el script con los argumentos
done

# Mostrar las sesiones activas
tmux list-sessions

echo "Sesiones de tmux creadas y scripts ejecutándose:"
for i in {0..19}; do
    echo "  - ${SESSION_BASE}_${i} ejecutando $SCRIPT con clase $i, modelo $MODEL y sensor $SENSOR"
done
