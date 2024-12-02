import cv2
import numpy as np

def superponer_png_sobre_jpg(imagen_fondo_path, imagen_png_path, salida_path):
    # Cargar la imagen de fondo (JPG)
    fondo = cv2.imread(imagen_fondo_path, cv2.IMREAD_COLOR)

    # Cargar la imagen PNG con transparencia (4 canales: BGR + Alfa)
    objeto_transparente = cv2.imread(imagen_png_path, cv2.IMREAD_UNCHANGED)

    # Separar los canales RGBA
    b, g, r, alpha = cv2.split(objeto_transparente)

    # Normalizar el canal alfa (0 a 1)
    alpha = alpha / 255.0

    # Combinar el objeto transparente con el fondo
    for c in range(3):  # Iterar sobre canales BGR
        fondo[:, :, c] = fondo[:, :, c] * (1 - alpha) + (objeto_transparente[:, :, c] * alpha)

    # Guardar la imagen combinada
    cv2.imwrite(salida_path, fondo)

    return salida_path

# Archivos de entrada y salida
imagen_fondo_path = 'data/back_1_davis346.jpg'  # Imagen JPG (fondo)
imagen_png_path = 'data/objeto_transparente_revisado.png'  # Imagen PNG transparente
salida_path = 'data/objeto_transparente_revisado_davis346.jpg'  # Salida

# Ejecutar la función
resultado = superponer_png_sobre_jpg(imagen_fondo_path, imagen_png_path, salida_path)
