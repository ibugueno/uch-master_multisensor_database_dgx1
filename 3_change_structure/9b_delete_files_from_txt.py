import os

def delete_files_from_txt(root_path, txt_file):
    """
    Elimina archivos presentes en un root_path que están listados en un archivo .txt,
    reemplazando la extensión .jpg por .txt.

    Args:
        root_path (str): Ruta raíz desde donde se eliminarán los archivos.
        txt_file (str): Archivo .txt que contiene las rutas relativas de los archivos a eliminar.
    """
    # Leer las rutas relativas desde el archivo .txt y reemplazar .jpg por .txt
    with open(txt_file, 'r') as f:
        files_to_delete = [line.strip().replace('.jpg', '.txt') for line in f.readlines()]
    
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


root_paths = ['output/labels-events-2/detection/yolo', 'output/labels-events-2/pose6d']
sensors = ['davis346', 'evk4']

for root_path in root_paths:
    for sensor in sensors:

        # Ejemplo de uso
        txt_file = f'data/{sensor}_missing_files.txt'  # Especifica el archivo .txt con las rutas relativas

        delete_files_from_txt(root_path, txt_file)
