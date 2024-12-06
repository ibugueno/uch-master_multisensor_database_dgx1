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

# Define las rutas a las particiones de imágenes y etiquetas
root_path = '../../..'
exp_path = '1_without_back_without_blur'
sensor = 'asus'


train_images = f"{root_path}/output/{exp_path}/train/{sensor}"
train_labels = f"{root_path}/output/labels/detection/yolo/train/{sensor}"
val_images = f"{root_path}/output/{exp_path}/val/{sensor}"
val_labels = f"{root_path}/output/labels/detection/yolo/val/{sensor}"
test_images = f"{root_path}/output/{exp_path}/test/{sensor}"
test_labels = f"{root_path}/output/labels/detection/yolo/test/{sensor}"

# Configura el archivo YAML
data_config = {
    'path': '.',  # Ruta base, si todas las rutas son relativas a este directorio
    'train': train_images,
    'val': val_images,
    'test': test_images,
    'names': class_names
}

# Guarda la configuración en un archivo YAML
with open('1_data_config.yaml', 'w') as file:
    yaml.dump(data_config, file, default_flow_style=False)

print("Archivo 'data_config.yaml' generado correctamente.")
