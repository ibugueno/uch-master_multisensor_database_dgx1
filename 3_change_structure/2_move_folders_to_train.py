import os
import shutil

def process_and_move_images(txt_file, output_folder, path_to_remove="", sensor=""):
    """
    Lee un archivo .txt con rutas a imágenes, reemplaza los '/' en las rutas por '_',
    elimina una parte específica de la ruta si se proporciona, y mueve las imágenes
    a una carpeta de salida según criterios específicos.

    Args:
        txt_file (str): Ruta al archivo .txt con rutas de imágenes.
        output_folder (str): Carpeta de destino para las imágenes con nuevos nombres.
        path_to_remove (str): Prefijo de la ruta que se debe eliminar del nombre.
        sensor (str): Nombre del sensor que se debe manejar especialmente en el nuevo nombre.
    """
    # Crear las subcarpetas si no existen
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    test_folder = os.path.join(output_folder, "test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Listas de criterios para clasificar las imágenes
    train_orientations = [
        "orientation_9_-16_-12/",
        "orientation_2_-87_19/",
        "orientation_-155_-15_102/",
        "orientation_-96_-7_-115/",
        "orientation_30_-17_-126/",
        "orientation_-145_-36_-162/",
        "orientation_-139_17_153/",
        "orientation_168_-58_-68/",
        "orientation_39_17_-102/",
        "orientation_-125_66_-116/",
        "orientation_-131_-26_-66/",
        "orientation_-23_16_-58/",
        "orientation_30_-59_-84/",
        "orientation_-36_-42_-170/",
        "orientation_-137_70_-80/",
        "orientation_145_25_11/",
        "orientation_-77_54_173/",
        "orientation_20_3_0/",
        "orientation_19_31_21/",
        "orientation_88_-6_-34/"
    ]

    val_orientations = [
        "orientation_-113_-15_75/",
        "orientation_-10_72_-129/"
    ]

    test_orientations = [
        "orientation_-159_-41_-21/",
        "orientation_-28_0_6/"
    ]

    # Leer el archivo .txt
    with open(txt_file, "r") as f:
        image_paths = f.readlines()
    
    for image_path in image_paths:
        image_path = image_path.strip()  # Quita espacios o saltos de línea
        
        # Verificar si la imagen existe
        if not os.path.exists(image_path):
            print(f"Archivo no encontrado: {image_path}")
            continue

        # Eliminar el prefijo especificado de la ruta, si está presente
        clean_path = image_path
        if path_to_remove and image_path.startswith(path_to_remove):
            clean_path = image_path[len(path_to_remove):]

        # Determinar la carpeta de destino basada en el nombre
        destination_folder = None
        if any(orientation in image_path for orientation in train_orientations):
            destination_folder = train_folder
        elif any(orientation in image_path for orientation in val_orientations):
            destination_folder = val_folder
        elif any(orientation in image_path for orientation in test_orientations):
            destination_folder = test_folder

        # Si no coincide con ningún criterio, omitir el archivo
        if destination_folder is None:
            print(f"Archivo no clasificado: {image_path}")
            continue

        # Crear el nuevo nombre reemplazando '/' por '_'
        new_name = clean_path.replace("/", "_")
        new_name = new_name.lstrip("_")  # Eliminar cualquier '_' inicial

        # Reemplazar el primer "_" después del sensor por "/"
        if sensor in new_name:
            new_name = new_name.replace(f"{sensor}_", f"{sensor}/", 1)

        # Crear la ruta completa para la nueva ubicación
        new_path = os.path.join(destination_folder, new_name)

        # Crear directorio en la nueva ubicación si no existe
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Mover la imagen al nuevo destino
        shutil.move(image_path, new_path)
        print(f"Imagen movida: {image_path} -> {new_path}")

# Especifica las rutas
sensor = 'evk4'
path = '4_with_back_noisy'
txt_file = f'data/output_ddbb-s-events_{path}_{sensor}_files.txt'  # Ruta al archivo .txt con rutas de imágenes
output_folder = f"output/ddbb-s-events/{path}"          # Carpeta de destino para las imágenes renombradas
path_to_remove = f"output/ddbb-s-events/{path}"         # Parte de la ruta que se debe eliminar

# Ejecutar la función
process_and_move_images(txt_file, output_folder, path_to_remove, sensor)
