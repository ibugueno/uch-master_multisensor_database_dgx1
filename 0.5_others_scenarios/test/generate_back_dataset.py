import cv2
import os
import numpy as np

def eliminar_fondo(objeto_path, color_fondo, porcentaje_tolerancia):
    # Cargar la imagen
    objeto = cv2.imread(objeto_path, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen tiene canal alfa; si no, añadirlo
    if objeto.shape[2] == 3:  # Si no tiene canal alfa
        objeto = cv2.cvtColor(objeto, cv2.COLOR_BGR2BGRA)

    # Convertir color de fondo a BGR
    color_fondo_bgr = tuple(reversed(color_fondo))  # Convertir de RGB a BGR

    # Calcular tolerancia
    tolerancia = int(255 * (porcentaje_tolerancia / 100.0))

    # Crear rangos de color (en BGR directamente)
    lower_bgr = np.array([max(0, c - tolerancia) for c in color_fondo_bgr], dtype=np.uint8)
    upper_bgr = np.array([min(255, c + tolerancia) for c in color_fondo_bgr], dtype=np.uint8)

    # Crear una máscara que identifique el fondo
    mascara = cv2.inRange(objeto[:, :, :3], lower_bgr, upper_bgr)

    # Invertir la máscara para conservar solo el objeto
    mascara_invertida = cv2.bitwise_not(mascara)

    # Aplicar la máscara al canal alfa
    objeto[:, :, 3] = mascara_invertida

    return objeto

def superponer_png_sobre_jpg(imagen_fondo_path, objeto_transparente, salida_path):
    # Cargar la imagen de fondo (JPG)
    fondo = cv2.imread(imagen_fondo_path, cv2.IMREAD_COLOR)

    # Separar los canales RGBA del objeto transparente
    b, g, r, alpha = cv2.split(objeto_transparente)

    # Normalizar el canal alfa (0 a 1)
    alpha = alpha / 255.0

    # Combinar el objeto transparente con el fondo
    for c in range(3):  # Iterar sobre canales BGR
        fondo[:, :, c] = fondo[:, :, c] * (1 - alpha) + (objeto_transparente[:, :, c] * alpha)

    # Guardar la imagen combinada
    cv2.imwrite(salida_path, fondo)

def procesar_imagenes(directorio_objetos, imagen_fondo_path, directorio_salida, color_fondo, porcentaje_tolerancia):
    # Crear el directorio de salida si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    # Recorrer todas las imágenes en el directorio de objetos
    for nombre_archivo in os.listdir(directorio_objetos):
        if nombre_archivo.endswith('.png'):  # Procesar solo archivos JPG
            objeto_path = os.path.join(directorio_objetos, nombre_archivo)
            salida_path = os.path.join(directorio_salida, nombre_archivo)

            # Eliminar el fondo del objeto
            objeto_transparente = eliminar_fondo(objeto_path, color_fondo, porcentaje_tolerancia)

            # Superponer el objeto transparente sobre el fondo conocido
            superponer_png_sobre_jpg(imagen_fondo_path, objeto_transparente, salida_path)

            print(f'Procesado: {nombre_archivo}')

# Parámetros
directorio_objetos = 'data/almohada'  # Directorio con imágenes de objetos
imagen_fondo_path = 'data/back_1_davis346.jpg'  # Fondo conocido
directorio_salida = 'data/salida'  # Directorio para guardar las imágenes procesadas
#color_fondo = (92, 79, 79)  # Color del fondo a eliminar (RGB)
color_fondo = (71, 71, 71)  # Color del fondo a eliminar (RGB)
porcentaje_tolerancia = 10  # Porcentaje de tolerancia para el color de fondo

# Ejecutar el procesamiento
procesar_imagenes(directorio_objetos, imagen_fondo_path, directorio_salida, color_fondo, porcentaje_tolerancia)
