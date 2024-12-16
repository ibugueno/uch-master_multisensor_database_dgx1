def find_extra_files(first_file, second_file, output_file):
    """
    Identifica archivos que están en la segunda lista pero no en la primera.

    Args:
        first_file (str): Ruta al archivo .txt con las rutas relativas de la primera carpeta.
        second_file (str): Ruta al archivo .txt con las rutas relativas de la segunda carpeta.
        output_file (str): Ruta al archivo .txt donde se guardarán las rutas sobrantes.
    """
    # Leer los archivos de entrada
    with open(first_file, "r") as f1:
        first_set = set(line.strip() for line in f1)
    
    with open(second_file, "r") as f2:
        second_set = set(line.strip() for line in f2)

    # Identificar los archivos sobrantes en la segunda carpeta
    extra_files = second_set - first_set

    # Guardar las rutas sobrantes en el archivo de salida
    with open(output_file, "w") as f:
        for file in sorted(extra_files):  # Opcional: ordenar las rutas
            f.write(file + "\n")

    print(f"Se encontraron {len(extra_files)} archivos sobrantes. Guardados en '{output_file}'")

# Rutas de los archivos .txt generados anteriormente
first_output_file = "data/labels_detection_yolo_files.txt"
second_output_file = "data/labels_pose6d_files.txt"
extra_files_output = "data/labels_pose6d_extra_files.txt"

# Ejecutar la función
find_extra_files(first_output_file, second_output_file, extra_files_output)
