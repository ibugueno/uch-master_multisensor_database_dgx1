import os

def list_txt_files(directory, output_file):
    """
    Lista todos los archivos .txt en un directorio y sus subdirectorios y guarda las rutas en un archivo.

    Args:
        directory (str): Ruta al directorio base.
        output_file (str): Ruta al archivo de salida donde se guardarán las rutas.
    
    Returns:
        set: Conjunto de rutas relativas de archivos .txt encontrados.
    """
    txt_files = set()
    with open(output_file, "w") as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    # Ruta relativa para comparar fácilmente entre carpetas
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    txt_files.add(relative_path)
                    f.write(relative_path + "\n")
    return txt_files

# Ruta de la primera carpeta
first_folder = "output/labels/detection/yolo"
first_output_file = "data/labels_detection_yolo_files.txt"

# Ruta de la segunda carpeta
second_folder = "output/labels/pose6d"
second_output_file = "data/labels_pose6d_files.txt"

# Listar los archivos .txt de ambas carpetas y guardar las rutas en archivos
first_folder_txt_files = list_txt_files(first_folder, first_output_file)
second_folder_txt_files = list_txt_files(second_folder, second_output_file)

# Mostrar los resultados
print(f"Archivos en '{first_folder}' guardados en '{first_output_file}': {len(first_folder_txt_files)} encontrados")
print(f"Archivos en '{second_folder}' guardados en '{second_output_file}': {len(second_folder_txt_files)} encontrados")
