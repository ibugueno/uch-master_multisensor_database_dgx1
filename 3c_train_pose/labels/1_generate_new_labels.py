import os

# Rutas de los archivos de entrada
yolo_files_list = "data/ddbb-s-frames_detection_yolo_files.txt"  # Lista de archivos YOLO
pose6d_files_list = "data/ddbb-s-frames_pose6d_files.txt"  # Lista de archivos Pose6D

output_dir = "../output/ddbb-s-frames/labels/pose6d-fixed"  # Carpeta base para los nuevos archivos combinados

yolo_prefix = "../output/ddbb-s-frames/labels/detection/yolo/"  # Prefijo para los archivos YOLO
pose6d_prefix = "../output/ddbb-s-frames/labels/pose6d/"  # Prefijo para los archivos Pose6D

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Leer las rutas de los archivos desde las listas
with open(yolo_files_list, 'r') as yolo_list_file:
    yolo_files = [os.path.join(yolo_prefix, line.strip()) for line in yolo_list_file.readlines()]

with open(pose6d_files_list, 'r') as pose6d_list_file:
    pose6d_files = [os.path.join(pose6d_prefix, line.strip()) for line in pose6d_list_file.readlines()]

'''
# Verificar que ambas listas tengan la misma cantidad de archivos
if len(yolo_files) != len(pose6d_files):
    raise ValueError("La cantidad de archivos en las listas de YOLO y Pose6D no coincide.")


# Verificar correspondencia 1 a 1
discrepancies = []

for yolo_file, pose6d_file in zip(yolo_files, pose6d_files):
    if not os.path.exists(yolo_file):
        discrepancies.append(f"Archivo YOLO no encontrado: {yolo_file}")
    if not os.path.exists(pose6d_file):
        discrepancies.append(f"Archivo Pose6D no encontrado: {pose6d_file}")

if discrepancies:
    print("Discrepancias encontradas:")
    for discrepancy in discrepancies:
        print(discrepancy)
    raise ValueError("Existen archivos faltantes o no emparejados. Corrige las discrepancias antes de continuar.")
'''

# Procesar cada par de archivos
for yolo_file, pose6d_file in zip(yolo_files, pose6d_files):
    # Leer contenido del archivo YOLO
    with open(yolo_file, 'r') as yolo_f:
        yolo_lines = yolo_f.readlines()

    # Leer contenido del archivo Pose6D
    with open(pose6d_file, 'r') as pose6d_f:
        pose6d_lines = pose6d_f.readlines()

    # Determinar la ruta relativa del archivo Pose6D para replicar la estructura
    relative_path = os.path.relpath(pose6d_file, pose6d_prefix)
    output_file_path = os.path.join(output_dir, relative_path)

    # Crear los directorios necesarios en la salida
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Escribir el archivo combinado en la ruta de salida
    with open(output_file_path, 'w') as out_f:
        for yolo_line, pose6d_line in zip(yolo_lines, pose6d_lines):
            # Parsear datos de YOLO
            yolo_parts = yolo_line.strip().split()
            x_cen = float(yolo_parts[1])  # Centro X (normalizado)
            y_cen = float(yolo_parts[2])  # Centro Y (normalizado)
            width = float(yolo_parts[3])  # Ancho (normalizado)
            height = float(yolo_parts[4])  # Alto (normalizado)

            # Parsear datos de Pose6D
            pose6d_parts = pose6d_line.strip().split(',')
            id_clase = int(pose6d_parts[0])  # ID de clase
            z_distance = float(pose6d_parts[3])  # Distancia
            q1 = float(pose6d_parts[4])  # Cuaternión q1
            q2 = float(pose6d_parts[5])  # Cuaternión q2
            q3 = float(pose6d_parts[6])  # Cuaternión q3
            q4 = float(pose6d_parts[7])  # Cuaternión q4

            # Crear la línea combinada
            combined_line = f"{id_clase},{x_cen:.6f},{y_cen:.6f},{width:.6f},{height:.6f},{z_distance:.6f},{q1:.6f},{q2:.6f},{q3:.6f},{q4:.6f}\n"
            out_f.write(combined_line)

print(f"Archivos combinados generados en la carpeta: {output_dir}")
