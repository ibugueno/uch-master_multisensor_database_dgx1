import cv2
import numpy as np

# Cargar las imágenes
objeto = cv2.imread('data/almohada/image_0500.png', cv2.IMREAD_UNCHANGED)


# Verificar si la imagen tiene canal alfa; si no, añadirlo
if objeto.shape[2] == 3:  # Si no tiene canal alfa
    objeto = cv2.cvtColor(objeto, cv2.COLOR_BGR2BGRA)

# Color de fondo a eliminar (en RGB, convertir a BGR para OpenCV)
color_fondo = (92, 79, 79)  # Color del fondo (R, G, B -> ajustar en BGR)
color_fondo_bgr = tuple(reversed(color_fondo))  # Convertir a BGR

# Tolerancia para la similaridad (porcentaje)
porcentaje_tolerancia = 10  # Cambiar según la necesidad
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

# Guardar el resultado como una imagen PNG transparente
cv2.imwrite('data/objeto_transparente_revisado.png', objeto)

# Mostrar la imagen procesada (opcional)
'''
cv2.imshow('Objeto Transparente', objeto)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''