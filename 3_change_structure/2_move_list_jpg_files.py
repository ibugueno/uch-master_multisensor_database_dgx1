import os

def list_jpg_files(directory, output_file):
    """
    Lista todos los archivos .jpg en un directorio y sus subdirectorios
    y guarda las rutas en un archivo .jpg.
    
    Args:
        directory (str): Ruta al directorio base.
        output_file (str): Ruta al archivo .jpg donde se guardarán las rutas.
    """
    jpg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):  # Ignora mayúsculas/minúsculas
                jpg_files.append(os.path.join(root, file))
    
    # Guardar las rutas en un archivo .jpg
    with open(output_file, "w") as f:
        for file in jpg_files:
            f.write(file + "\n")
    
    print(f"Se encontraron {len(jpg_files)} archivos .jpg. Las rutas se han guardado en {output_file}")


# Ruta del directorio base
sensor = 'asus'
base_path = 'output/ddbs-s_labels_original/segmentation'
base_directory = f'{base_path}/{sensor}'

# Reemplazar '/' con '_' en base_path para la salida
sanitized_base_path = base_path.replace("/", "_").rstrip("_")
output_jpg = f'data/{sanitized_base_path}_{sensor}_files.txt'

# Llamada a la función
list_jpg_files(base_directory, output_jpg)
