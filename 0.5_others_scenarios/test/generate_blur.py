import cv2
import os

# Función para aplicar blur y guardar imágenes en un nuevo directorio
def apply_blur_to_images(input_dir, output_dir, blur_kernel=(31, 31)):
    # Verificar si el directorio de salida existe; si no, crearlo
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extensiones válidas
    valid_extensions = {".jpg", ".jpeg", ".png"}

    # Recorrer todas las imágenes en el directorio de entrada
    for filename in os.listdir(input_dir):
        # Obtener la extensión del archivo
        ext = os.path.splitext(filename)[1].lower()
        
        # Procesar solo archivos con extensiones válidas
        if ext in valid_extensions:
            input_path = os.path.join(input_dir, filename)
            
            # Leer la imagen
            image = cv2.imread(input_path)
            
            if image is not None:
                # Aplicar blur
                blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=10, sigmaY=10)
                
                # Guardar la imagen procesada en el directorio de salida
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, blurred_image)
                print(f"Procesada y guardada: {output_path}")
            else:
                print(f"El archivo no es una imagen válida o está corrupto: {filename}")
        else:
            print(f"Formato no válido, se omite: {filename}")

# Ejemplo de uso
input_directory = 'data/zapatilla'
output_directory = 'data/zapatilla_blur'
apply_blur_to_images(input_directory, output_directory, blur_kernel=(5, 5))
