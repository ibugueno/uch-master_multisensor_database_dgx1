import cv2
import os
import numpy as np

# Diccionario de clases con índices
clases = {
    'almohada': 0,
    'arbol': 1,
    'avion': 2,
    'boomerang': 3,
    'caja_amarilla': 4,
    'caja_azul': 5,
    'carro_rojo': 6,
    'clorox': 7,
    'dino': 8,
    'disco': 9,
    'jarron': 10,
    'lysoform': 11,
    'mobil': 12,
    'paleta': 13,
    'pelota': 14,
    'sombrero': 15,
    'tarro': 16,
    'tazon': 17,
    'toalla_roja': 18,
    'zapatilla': 19
}

def encontrar_clase_en_path(path):
    """
    Busca si alguno de los nombres de las clases está en cualquier parte del path.
    """
    for nombre_clase, indice_clase in clases.items():
        if nombre_clase in path:
            return indice_clase
    return None

def seleccionar_fondo(directorio_objetos):
    """
    Selecciona el fondo adecuado según el nombre del directorio.
    """
    if "evk4" in directorio_objetos:
        return 'back/back_1_evk4.jpg'
    elif "davis346" in directorio_objetos:
        return 'back/back_1_davis346.jpg'
    elif "asus" in directorio_objetos:
        return 'back/back_1_asus.jpg'
    elif "zed2" in directorio_objetos:
        return 'back/back_1_zed2.jpg'
    else:
        raise ValueError("No se reconoce el tipo de fondo para el directorio proporcionado.")

def eliminar_fondo(objeto_path, color_fondo, porcentaje_tolerancia):
    """
    Elimina el fondo de la imagen basándose en el color y la tolerancia especificada.
    """
    objeto = cv2.imread(objeto_path, cv2.IMREAD_UNCHANGED)
    if objeto.shape[2] == 3:
        objeto = cv2.cvtColor(objeto, cv2.COLOR_BGR2BGRA)
    color_fondo_bgr = tuple(reversed(color_fondo))
    tolerancia = int(255 * (porcentaje_tolerancia / 100.0))
    lower_bgr = np.array([max(0, c - tolerancia) for c in color_fondo_bgr], dtype=np.uint8)
    upper_bgr = np.array([min(255, c + tolerancia) for c in color_fondo_bgr], dtype=np.uint8)
    mascara = cv2.inRange(objeto[:, :, :3], lower_bgr, upper_bgr)
    mascara_invertida = cv2.bitwise_not(mascara)
    objeto[:, :, 3] = mascara_invertida
    return objeto, mascara_invertida

def aplicar_morfologia(mascara, kernel_size=5):
    """
    Aplica operaciones morfológicas para refinar la máscara.
    """
    kernel_er = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mascara_erodida = cv2.erode(mascara, kernel_er, iterations=1)
    mascara_dilatada = cv2.dilate(mascara_erodida, kernel_dil, iterations=1)
    return mascara_dilatada

def obtener_bounding_box(mascara):
    """
    Obtiene el bounding box del objeto más grande en la máscara.
    """
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
        return x, y, w, h
    return None

def superponer_png_sobre_jpg(imagen_fondo_path, objeto_transparente):
    """
    Superpone un PNG con canal alfa sobre un fondo JPG.
    """
    fondo = cv2.imread(imagen_fondo_path, cv2.IMREAD_COLOR)
    b, g, r, alpha = cv2.split(objeto_transparente)
    alpha = alpha / 255.0
    for c in range(3):  # Iterar sobre canales BGR
        fondo[:, :, c] = fondo[:, :, c] * (1 - alpha) + objeto_transparente[:, :, c] * alpha
    return fondo

def replicar_estructura_y_guardar(base_salida, relative_path, contenido, extension='.jpg'):
    """
    Guarda contenido en `base_salida` asegurando que los directorios necesarios existen,
    replicando la estructura relativa.
    """
    if extension:
        relative_path = os.path.splitext(relative_path)[0] + extension  # Cambiar la extensión si es necesario
    salida_path = os.path.join(base_salida, relative_path)
    os.makedirs(os.path.dirname(salida_path), exist_ok=True)  # Crear directorios si no existen
    if isinstance(contenido, np.ndarray):  # Si es una imagen
        cv2.imwrite(salida_path, contenido)
    elif isinstance(contenido, str):  # Si es texto (anotaciones)
        with open(salida_path, "w") as f:
            f.write(contenido)
    return salida_path

def procesar_imagenes_desde_txt(archivo_txt, directorio_salida, color_fondo, porcentaje_tolerancia, nombres_salida, prefijo=""):
    """
    Procesa las imágenes desde un archivo .txt con rutas, eliminando fondo y generando salidas.
    Si no se encuentra un objeto, se almacena solo el fondo.
    """
    with open(archivo_txt, 'r') as file:
        rutas_imagenes = [os.path.join(prefijo, line.strip()) for line in file if line.strip()]

    total_imagenes = len(rutas_imagenes)
    procesadas = 0
    no_procesadas = 0

    for objeto_path in rutas_imagenes:
        print(f"[DEBUG] Procesando archivo: {objeto_path}")
        try:
            # Seleccionar fondo basado en el directorio o archivo
            imagen_fondo_path = seleccionar_fondo(objeto_path)
            indice_clase = encontrar_clase_en_path(objeto_path)
            if indice_clase is None:
                print(f"[DEBUG] No se encontró una clase en el path: {objeto_path}")
                no_procesadas += 1
                continue

            # Procesar la imagen y eliminar fondo
            objeto_transparente, mascara = eliminar_fondo(objeto_path, color_fondo, porcentaje_tolerancia)
            mascara_refinada = aplicar_morfologia(mascara)
            bounding_box = obtener_bounding_box(mascara_refinada)

            # Relative path para almacenar las salidas
            relative_path = os.path.relpath(objeto_path, prefijo)

            if bounding_box:
                # Coordenadas YOLO y RCNN
                x, y, w, h = bounding_box
                x_centro = (x + w / 2) / mascara.shape[1]
                y_centro = (y + h / 2) / mascara.shape[0]
                ancho = w / mascara.shape[1]
                alto = h / mascara.shape[0]

                x_min, y_min, x_max, y_max = x, y, x + w, y + h

                # Guardar salidas estándar
                '''
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("segmentacion", "ddbs-s_labels/segmentation")),
                    relative_path,
                    mascara_refinada,
                    ".jpg"
                )
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("anotaciones_yolo", "ddbs-s_labels/detection/yolo")),
                    relative_path,
                    f"{indice_clase} {x_centro:.6f} {y_centro:.6f} {ancho:.6f} {alto:.6f}\n",
                    ".txt"
                )
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("anotaciones_rcnn", "ddbs-s_labels/detection/rcnn")),
                    relative_path,
                    f"{indice_clase},{x_min},{y_min},{x_max},{y_max}\n",
                    ".txt"
                )
                # Generar imagen sin blur
                imagen_sin_blur = superponer_png_sobre_jpg(imagen_fondo_path, objeto_transparente)
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("sin_blur", "3_ddbs-s_with_back_without_blur")),
                    relative_path,
                    imagen_sin_blur
                )

                # Generar imagen con blur
                objeto_desenfocado = objeto_transparente.copy()
                objeto_desenfocado[:, :, :3] = cv2.GaussianBlur(objeto_transparente[:, :, :3], (7, 7), 10)
                imagen_con_blur = superponer_png_sobre_jpg(imagen_fondo_path, objeto_desenfocado)
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("con_blur", "4_ddbs-s_with_back_with_blur")),
                    relative_path,
                    imagen_con_blur
                )
                '''
                procesadas += 1
            else:
                # No se encontró bounding box, guardar solo el fondo
                print(f"[DEBUG] No se encontró un objeto en: {objeto_path}. Se almacenará solo el fondo.")

                # Guardar fondo sin blur
                fondo = cv2.imread(imagen_fondo_path)
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("sin_blur", "3_ddbs-s_with_back_without_blur")),
                    relative_path,
                    fondo
                )

                # Guardar fondo con blur
                fondo_blur = cv2.GaussianBlur(fondo, (7, 7), 10)
                replicar_estructura_y_guardar(
                    os.path.join(directorio_salida, nombres_salida.get("con_blur", "4_ddbs-s_with_back_with_blur")),
                    relative_path,
                    fondo
                )
                no_procesadas += 1

        except Exception as e:
            print(f"[ERROR] Error procesando {objeto_path}: {e}")
            no_procesadas += 1

    # Resumen final
    print(f"\n[RESUMEN] Total imágenes: {total_imagenes}")
    print(f"[RESUMEN] Procesadas con objetos: {procesadas}")
    print(f"[RESUMEN] Sin objetos (solo fondo): {no_procesadas}")

# Parámetros
archivo_rutas = 'data/rutas_corregidas.txt'
directorio_salida = 'output/'
color_fondo = (71, 71, 71)
porcentaje_tolerancia = 2
prefijo = "input/"
nombres_salida = {
    "sin_blur": "3_ddbs-s_with_back_without_blur",
    "con_blur": "4_ddbs-s_with_back_with_blur",
    "segmentacion": "ddbs-s_labels/segmentation",
    "anotaciones_yolo": "ddbs-s_labels/detection/yolo",
    "anotaciones_rcnn": "ddbs-s_labels/detection/rcnn"
}

procesar_imagenes_desde_txt(archivo_rutas, directorio_salida, color_fondo, porcentaje_tolerancia, nombres_salida, prefijo)
