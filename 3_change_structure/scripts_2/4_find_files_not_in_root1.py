import os

def list_jpg_files(root_path):
    """
    Lista recursivamente todos los archivos .jpg en una ruta dada con sus rutas relativas.

    Args:
        root_path (str): Ruta raíz desde donde buscar los archivos.

    Returns:
        set: Conjunto de rutas relativas de archivos .jpg.
    """
    jpg_files = set()
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith('.jpg'):
                relative_path = os.path.relpath(os.path.join(dirpath, file), root_path)
                jpg_files.add(relative_path)
    return jpg_files

def find_files_not_in_root1(root_path_1, root_path_2, output_file):
    """
    Genera un archivo .txt con las rutas relativas de los archivos .jpg que están en root_path_2
    pero no en root_path_1.

    Args:
        root_path_1 (str): Primera ruta raíz.
        root_path_2 (str): Segunda ruta raíz.
        output_file (str): Ruta al archivo de salida.
    """
    # Obtener las rutas relativas de los archivos .jpg en ambas rutas
    jpg_files_1 = list_jpg_files(root_path_1)
    jpg_files_2 = list_jpg_files(root_path_2)

    # Encontrar los archivos que están en root_path_2 pero no en root_path_1
    difference_files = jpg_files_2 - jpg_files_1

    # Escribir los archivos encontrados en el archivo de salida
    with open(output_file, 'w') as f:
        for file in sorted(difference_files):  # Ordenar para una salida más organizada
            f.write(file + '\n')

    print(f"Se generó el archivo {output_file} con {len(difference_files)} entradas.")

# Ejemplo de uso
root_path_1 = 'output/ddbb-s-events/1_without_back_clean/'  # Ruta a la primera carpeta
root_path_2 = 'output/ddbb-s-events/2_without_back_noisy/'  # Ruta a la segunda carpeta
output_file = 'data/output_not_in_root.txt'  # Archivo de salida

find_files_not_in_root1(root_path_1, root_path_2, output_file)
