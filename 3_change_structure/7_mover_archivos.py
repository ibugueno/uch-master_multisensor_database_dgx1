import os
import shutil

def move_files_with_structure(input_txt, source_prefix, output_root):
    """
    Mueve archivos listados en un archivo .txt a un nuevo directorio respetando su estructura.

    Args:
        input_txt (str): Ruta al archivo .txt con rutas relativas de los archivos.
        source_prefix (str): Prefijo para construir las rutas completas de los archivos originales.
        output_root (str): Directorio raíz donde se copiarán los archivos manteniendo su estructura.
    """
    # Leer las rutas relativas desde el archivo
    with open(input_txt, "r") as f:
        relative_paths = [line.strip() for line in f if line.strip()]

    for relative_path in relative_paths:
        # Construir la ruta completa al archivo original
        source_file = os.path.join(source_prefix, relative_path)

        # Construir la ruta de destino respetando la estructura relativa
        destination_file = os.path.join(output_root, relative_path)

        # Crear los directorios en la ruta de destino si no existen
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)

        # Mover el archivo al destino
        if os.path.exists(source_file):
            shutil.move(source_file, destination_file)
            print(f"Archivo movido: {source_file} -> {destination_file}")
        else:
            print(f"Archivo no encontrado: {source_file}")

# Rutas a los archivos y directorios
input_txt = "data/labels_pose6d_extra_files.txt"  # Archivo .txt con las rutas relativas
source_prefix = "output/labels/pose6d"            # Prefijo para las rutas originales
output_root = "output/labels/pose6d_structure"    # Directorio de destino

# Ejecutar la función
move_files_with_structure(input_txt, source_prefix, output_root)
