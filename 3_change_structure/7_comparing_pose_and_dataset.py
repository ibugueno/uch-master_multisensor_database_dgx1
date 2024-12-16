import os

def list_jpg_files_with_filter(root_path, output_txt, filter_word="evk4"):
    """
    Lista todos los archivos .jpg en una carpeta y sus subcarpetas que contienen una palabra específica en su ruta,
    guardándolos en un archivo .txt.

    Args:
        root_path (str): Carpeta raíz donde buscar los archivos .jpg.
        output_txt (str): Ruta del archivo .txt donde se guardarán las rutas relativas.
        filter_word (str): Palabra que debe estar presente en la ruta para incluir el archivo.
    """
    jpg_files = []
    
    # Recorrer la carpeta recursivamente
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".jpg"):
                relative_path = os.path.relpath(os.path.join(dirpath, filename), root_path)
                if filter_word in relative_path:
                    jpg_files.append(relative_path)
    
    # Guardar en un archivo .txt
    with open(output_txt, "w") as f:
        for file in jpg_files:
            f.write(file + "\n")
    
    print(f"Se guardaron {len(jpg_files)} archivos en {output_txt} que contienen la palabra '{filter_word}'")

sensor = 'evk4'

# Rutas de las carpetas y archivos .txt
root_path_1 = "output/labels-events/segmentation/"
output_txt_1 = f"data/{sensor}_{root_path_1.replace('/','_')}_files.txt"

root_path_2 = "output/ddbb-s-events/4_with_back_noisy/"
output_txt_2 = f"data/{sensor}_{root_path_2.replace('/','_')}_files.txt"

# Generar los listados con filtro 'evk4'
list_jpg_files_with_filter(root_path_1, output_txt_1, filter_word=sensor)
list_jpg_files_with_filter(root_path_2, output_txt_2, filter_word=sensor)
