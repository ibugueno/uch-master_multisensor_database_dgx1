import os

def process_txt_files(input_path, txt_file_with_paths, output_root):
    """
    Procesa un archivo que contiene rutas a otros .txt, genera un archivo image_0000.txt
    para cada línea del contenido, y lo guarda en la misma estructura de carpetas en otro directorio.

    Args:
        input_path (str): Ruta base para las rutas relativas en el archivo txt_file_with_paths.
        txt_file_with_paths (str): Ruta al archivo que contiene las rutas a otros archivos .txt.
        output_root (str): Directorio raíz donde se crearán los nuevos archivos.
    """
    # Leer las rutas a los archivos .txt
    with open(txt_file_with_paths, "r") as f:
        txt_paths = f.readlines()

    for txt_path in txt_paths:
        txt_path = txt_path.strip()  # Limpiar espacios y saltos de línea

        # Convertir la ruta relativa a una ruta completa
        full_txt_path = os.path.join(input_path, txt_path)

        if not os.path.exists(full_txt_path):
            print(f"Archivo no encontrado: {full_txt_path}")
            continue

        # Leer el contenido del archivo actual
        with open(full_txt_path, "r") as current_txt:
            lines = current_txt.readlines()

        # Asegurarse de que el archivo tiene un header válido
        if not lines[0].strip().startswith("frame,x,y,z,qx,qy,qz,qw"):
            print(f"Formato inválido en el archivo: {full_txt_path}")
            continue

        # Procesar cada línea después del header
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Extraer el número de frame y formatearlo como image_0000.txt
            frame_number = int(line.split(",")[0])
            new_file_name = f"image_{frame_number:04d}.txt"

            # Generar la nueva ruta relativa desde input_path
            relative_path = os.path.relpath(full_txt_path, input_path)
            new_dir = os.path.join(output_root, os.path.dirname(relative_path))
            os.makedirs(new_dir, exist_ok=True)

            # Generar la ruta completa para el nuevo archivo
            new_file_path = os.path.join(new_dir, new_file_name)

            # Escribir la línea en el nuevo archivo
            with open(new_file_path, "w") as output_file:
                output_file.write(line + "\n")

            print(f"Archivo creado: {new_file_path}")

# Especifica las rutas
input_txt = "data/asus_pose_labels.txt"  # Archivo que contiene las rutas a otros .txt
input_path = "input/"  # Ruta base para los archivos de entrada
output_directory = "output/pose"  # Directorio raíz de salida

# Ejecutar la función
process_txt_files(input_path, input_txt, output_directory)
