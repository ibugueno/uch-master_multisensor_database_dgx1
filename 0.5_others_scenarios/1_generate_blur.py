import cv2
import os

# Función para aplicar blur a las imágenes y mantener la estructura de carpetas
def apply_blur_to_folders(txt_file, input_prefix='input/', output_prefix='output/', blur_kernel=(7, 7)):
    # Leer las direcciones desde el archivo .txt
    with open(txt_file, 'r') as file:
        folder_list = [line.strip() for line in file if line.strip()]  # Eliminar líneas vacías

    # Recorrer cada carpeta listada
    for folder in folder_list:
        input_dir = os.path.join(input_prefix, folder)
        output_dir = os.path.join(output_prefix, folder)
        
        # Asegurarse de que la carpeta de salida tenga la misma estructura
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Procesar las imágenes en la carpeta
        process_images_in_folder(input_dir, output_dir, blur_kernel)

# Función para procesar imágenes en una carpeta
def process_images_in_folder(input_dir, output_dir, blur_kernel=(7, 7)):
    # Extensiones válidas
    valid_extensions = {".jpg", ".jpeg", ".png"}

    for root, _, files in os.walk(input_dir):  # os.walk para procesar subcarpetas si las hubiera
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                input_path = os.path.join(root, file)
                
                # Crear el directorio equivalente en la carpeta de salida
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                # Procesar la imagen
                image = cv2.imread(input_path)
                if image is not None:
                    # Aplicar blur
                    blurred_image = cv2.GaussianBlur(image, blur_kernel, sigmaX=10, sigmaY=10)
                    
                    # Guardar la imagen procesada
                    output_path = os.path.join(target_dir, file)
                    cv2.imwrite(output_path, blurred_image)
                    print(f"Procesada y guardada: {output_path}")
                else:
                    print(f"El archivo no es una imagen válida o está corrupto: {input_path}")

# Ejemplo de uso
txt_file_path = 'input/all_folders.txt'  # Archivo .txt con las direcciones a carpetas
apply_blur_to_folders(txt_file_path, input_prefix='input/', output_prefix='output/', blur_kernel=(7, 7))
