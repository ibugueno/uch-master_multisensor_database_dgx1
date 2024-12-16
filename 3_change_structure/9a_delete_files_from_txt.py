import os

def delete_files_from_txt(root_path, txt_file):
    """
    Elimina archivos presentes en un root_path que están listados en un archivo .txt.

    Args:
        root_path (str): Ruta raíz desde donde se eliminarán los archivos.
        txt_file (str): Archivo .txt que contiene las rutas relativas de los archivos a eliminar.
    """
    # Leer las rutas relativas desde el archivo .txt
    with open(txt_file, 'r') as f:
        files_to_delete = [line.strip() for line in f.readlines()]
    
    # Contador de archivos eliminados
    deleted_count = 0

    for relative_path in files_to_delete:
        # Crear la ruta completa del archivo
        full_path = os.path.join(root_path, relative_path)
        
        if os.path.exists(full_path):
            try:
                os.remove(full_path)  # Eliminar el archivo
                deleted_count += 1
                print(f"Archivo eliminado: {full_path}")
            except Exception as e:
                print(f"Error al eliminar {full_path}: {e}")
        else:
            print(f"Archivo no encontrado, no se puede eliminar: {full_path}")
    
    print(f"Se eliminaron {deleted_count} archivos de {len(files_to_delete)} listados en el archivo .txt.")

sensor = 'evk4'

# Ejemplo de uso
root_path = 'output/labels-events-2/segmentation'  # 1_without_back_clean, 2_without_back_noisy, Especifica el directorio raíz
txt_file = f'data/{sensor}_missing_files.txt'  # Especifica el archivo .txt con las rutas relativas

delete_files_from_txt(root_path, txt_file)
