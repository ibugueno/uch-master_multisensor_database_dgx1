import os

def list_aedat_files_to_txt(base_dir, output_file):
    """
    Lista las rutas relativas de todos los archivos .aedat en un directorio y sus subdirectorios,
    y las guarda en un archivo .txt.

    Args:
        base_dir (str): Directorio base donde buscar los archivos.
        output_file (str): Ruta al archivo .txt donde guardar las rutas encontradas.
    """
    aedat_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".aedat4"):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                aedat_files.append(relative_path)

    # Guardar las rutas en un archivo .txt
    with open(output_file, 'w') as txt_file:
        for file in aedat_files:
            txt_file.write(file + '\n')

    print(f"Archivo generado: {output_file}")


if __name__ == "__main__":
    # Especifica el directorio base y el archivo de salida
    sensor = 'evk4'
    model = 'noisy'
    
    base_directory = f"../output/aedat_val_with_back_without_blur/{model}/{sensor}/"
    
    output_txt = f"data/{sensor}_{model}_aedat_files_list.txt"

    # Generar el archivo de salida
    list_aedat_files_to_txt(base_directory, output_txt)
