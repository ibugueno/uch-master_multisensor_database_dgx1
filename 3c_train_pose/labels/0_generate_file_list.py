import os

def generate_file_list(input_dir, output_file):
    """
    Genera un archivo .txt con las rutas relativas de todos los archivos .txt en el directorio dado.

    Args:
        input_dir (str): Directorio de entrada donde buscar archivos .txt.
        output_file (str): Ruta del archivo de salida donde se guardarán las rutas relativas.
    """
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    relative_path = os.path.relpath(os.path.join(root, file), input_dir)
                    f.write(relative_path + '\n')

# Directorios y nombres de archivo de salida
input_dir_1 = "../output/ddbb-s-frames/labels/detection/yolo/"
output_file_1 = "data/ddbb-s-frames_detection_yolo_files.txt"

input_dir_2 = "../output/ddbb-s-frames/labels/pose6d/"
output_file_2 = "data/ddbb-s-frames_pose6d_files.txt"

# Generar los archivos
generate_file_list(input_dir_1, output_file_1)
generate_file_list(input_dir_2, output_file_2)

print(f"Archivos generados: {output_file_1} y {output_file_2}")
