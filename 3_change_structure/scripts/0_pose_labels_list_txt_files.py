import os

def list_txt_files(directory, output_file, base_path):
    """
    Lista todos los archivos .txt en un directorio y sus subdirectorios,
    guarda las rutas relativas (excluyendo base_path) en un archivo .txt.
    
    Args:
        directory (str): Ruta al directorio base.
        output_file (str): Ruta al archivo .txt donde se guardarán las rutas.
        base_path (str): Ruta base a excluir de las rutas guardadas.
    """
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".txt"):  # Ignora mayúsculas/minúsculas
                # Obtener ruta relativa desde base_path
                relative_path = os.path.relpath(os.path.join(root, file), base_path)
                txt_files.append(relative_path)
    
    # Guardar las rutas en un archivo .txt
    with open(output_file, "w") as f:
        for file in txt_files:
            f.write(file + "\n")
    
    print(f"Se encontraron {len(txt_files)} archivos .txt. Las rutas relativas se han guardado en {output_file}")

# Ruta del directorio base
sensor = 'asus'
base_path = 'input'
base_directory = f'{base_path}/{sensor}'

# Ruta del archivo de salida
output_txt = f'data/{sensor}_pose_labels.txt'

# Llamada a la función
list_txt_files(base_directory, output_txt, base_path)
