import os

def restructure_file_names(input_txt, root, output_root):
    """
    Reestructura los nombres de archivos en un archivo .txt, creando una nueva estructura de carpetas.

    Args:
        input_txt (str): Ruta al archivo .txt con rutas relativas de archivos .jpg.
        root (str): Ruta raíz que se concatena a las rutas relativas.
        output_root (str): Carpeta base donde se creará la nueva estructura.
    """
    with open(input_txt, "r") as infile:
        for line in infile:
            relative_path = line.strip()  # Quita espacios y saltos de línea

            # Construir la ruta completa usando la raíz
            full_path = os.path.join(root, relative_path)

            # Extraer prefijo de la ruta (train/, val/, test/)
            subfolder = os.path.dirname(relative_path).split("/")[0]

            # Obtener el nombre del archivo
            filename = os.path.basename(relative_path)

            # Extraer la primera palabra antes del primer _
            first_word = filename.split("_", 1)[0]

            # Eliminar la primera palabra y el guion bajo del nombre del archivo
            new_filename = filename[len(first_word) + 1:]

            # Crear la nueva ruta
            new_relative_path = os.path.join(subfolder, first_word, new_filename)
            new_full_path = os.path.join(output_root, new_relative_path)

            # Crear los directorios si no existen
            os.makedirs(os.path.dirname(new_full_path), exist_ok=True)

            # Mover el archivo a la nueva ubicación
            if os.path.exists(full_path):
                os.rename(full_path, new_full_path)
                #print(f"Archivo movido: {relative_path} -> {new_relative_path}")
            else:
                #print(f"Archivo no encontrado: {relative_path}")
                pass

# Especificar las rutas
input_txt = "data/list_labels_files.txt"      # Archivo con rutas relativas de los archivos .jpg
root = "output/labels/pose6d"           # Ruta raíz para acceder a los archivos originales
output_root = root           # Carpeta base para la nueva estructura

# Ejecutar la función
restructure_file_names(input_txt, root, output_root)
