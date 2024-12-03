import cv2

def recortar_y_redimensionar(imagen_path, salida_path, ancho=1280, alto=720):
    # Leer la imagen
    imagen = cv2.imread(imagen_path)

    # Obtener dimensiones actuales
    alto_original, ancho_original = imagen.shape[:2]

    # Recorte inicial para hacer zoom (manteniendo la relación de aspecto original)
    x_inicio = ancho_original // 4
    x_fin = ancho_original - x_inicio
    y_inicio = alto_original // 4
    y_fin = alto_original - y_inicio
    imagen_recortada_zoom = imagen[y_inicio:y_fin, x_inicio:x_fin]

    # Actualizar dimensiones después del recorte inicial
    alto_zoom, ancho_zoom = imagen_recortada_zoom.shape[:2]

    # Relación de aspecto deseada
    relacion_deseada = ancho / alto

    # Recorte final para ajustar a la relación deseada
    if ancho_zoom / alto_zoom > relacion_deseada:
        # Recortar ancho
        nuevo_ancho = int(alto_zoom * relacion_deseada)
        x_inicio = (ancho_zoom - nuevo_ancho) // 2
        imagen_recortada_aspecto = imagen_recortada_zoom[:, x_inicio:x_inicio + nuevo_ancho]
    else:
        # Recortar alto
        nuevo_alto = int(ancho_zoom / relacion_deseada)
        y_inicio = (alto_zoom - nuevo_alto) // 2
        imagen_recortada_aspecto = imagen_recortada_zoom[y_inicio:y_inicio + nuevo_alto, :]

    # Redimensionar la imagen recortada
    imagen_redimensionada = cv2.resize(imagen_recortada_aspecto, (ancho, alto), interpolation=cv2.INTER_AREA)

    # Guardar la imagen procesada
    cv2.imwrite(salida_path, imagen_redimensionada)
    return salida_path

# Archivos de entrada y salida
imagen1_path = '../data/back_1.jpg'
imagen1_salida = '../data/back_1_evk4.jpg'

# Procesar imagen 1
resultado1 = recortar_y_redimensionar(imagen1_path, imagen1_salida)