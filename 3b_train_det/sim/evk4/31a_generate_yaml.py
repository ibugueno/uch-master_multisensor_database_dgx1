from ultralytics import YOLO
import yaml

# Diccionario de clases proporcionado
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

# Convertir el diccionario a un formato compatible con YOLOv8
class_names = {v: k for k, v in clases.items()}

# Define las rutas a las particiones de imágenes
exp_path = '3_with_back_clean'
sensor = 'evk4'

train_dir = f"input/ddbb-s-events/{exp_path}/train/{sensor}"
val_dir = f"input/ddbb-s-events/{exp_path}/val/{sensor}"
test_dir = f"input/ddbb-s-events/{exp_path}/test/{sensor}"

# Configura el archivo YAML en el formato estándar de YOLOv8
data_config = {
    'path': f'/app/paths/ddbb-s-events/{exp_path}',  # Ruta base
    'train': f"train/{sensor}",  # Subdirectorio de entrenamiento
    'val': f"val/{sensor}",      # Subdirectorio de validación
    'test': f"test/{sensor}",    # Subdirectorio de prueba
    'names': class_names         # Diccionario de clases
}

# Guarda la configuración en un archivo YAML
with open(f'data/{sensor}_{exp_path[:1]}_data_config.yaml', 'w') as file:
    yaml.dump(data_config, file, default_flow_style=False)

print(f"Archivo '{sensor}_{exp_path[:1]}_data_config.yaml' generado correctamente.")
