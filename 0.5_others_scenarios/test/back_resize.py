import cv2

def recortar_y_redimensionar(imagen_path, salida_path, ancho=346, alto=260):
    # Leer la imagen
    imagen = cv2.imread(imagen_path)

    # Obtener dimensiones actuales
    alto_original, ancho_original = imagen.shape[:2]

    # Relación de aspecto deseada
    relacion_deseada = ancho / alto

    # Calcular nuevas dimensiones para recortar
    if ancho_original / alto_original > relacion_deseada:
        # Recortar ancho
        nuevo_ancho = int(alto_original * relacion_deseada)
        x_inicio = (ancho_original - nuevo_ancho) // 2
        imagen_recortada = imagen[:, x_inicio:x_inicio + nuevo_ancho]
    else:
        # Recortar alto
        nuevo_alto = int(ancho_original / relacion_deseada)
        y_inicio = (alto_original - nuevo_alto) // 2
        imagen_recortada = imagen[y_inicio:y_inicio + nuevo_alto, :]

    # Redimensionar la imagen recortada
    imagen_redimensionada = cv2.resize(imagen_recortada, (ancho, alto), interpolation=cv2.INTER_AREA)

    # Guardar la imagen procesada
    cv2.imwrite(salida_path, imagen_redimensionada)
    return salida_path

# Archivos de entrada y salida
imagen1_path = 'data/back_1.jpg'
imagen1_salida = 'data/back_1_davis346.jpg'

# Procesar imagen 1
resultado1 = recortar_y_redimensionar(imagen1_path, imagen1_salida)

resultado1
