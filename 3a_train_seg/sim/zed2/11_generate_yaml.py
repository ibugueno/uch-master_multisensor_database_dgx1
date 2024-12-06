import yaml
import os

def generate_yaml(output_file, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, num_classes, hyperparameters, class_mapping, image_size, output_dir):
    """
    Genera un archivo YAML para configuraciones de entrenamiento.

    Args:
        output_file (str): Nombre del archivo YAML a generar.
        train_image_dir (str): Ruta al directorio de imágenes de entrenamiento.
        train_mask_dir (str): Ruta al directorio de máscaras de entrenamiento.
        val_image_dir (str): Ruta al directorio de imágenes de validación.
        val_mask_dir (str): Ruta al directorio de máscaras de validación.
        num_classes (int): Número de clases (incluyendo fondo).
        hyperparameters (dict): Diccionario con los hiperparámetros.
        class_mapping (dict): Diccionario que mapea nombres de clases a sus IDs.
        image_size (tuple): Dimensiones de las imágenes (ancho, alto).
        output_dir (str): Carpeta donde se guardarán métricas y modelos.

    Returns:
        None
    """
    config = {
        'train': {
            'images': train_image_dir,
            'masks': train_mask_dir,
        },
        'val': {
            'images': val_image_dir,
            'masks': val_mask_dir,
        },
        'num_classes': num_classes,
        'hyperparameters': hyperparameters,
        'classes': class_mapping,
        'image_size': {
            'width': image_size[0],
            'height': image_size[1]
        },
        'output_dir': output_dir
    }

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Archivo YAML generado: {output_file}")

if __name__ == "__main__":
    # Rutas del dataset
    exp_path = '1_without_back_without_blur'
    sensor = 'zed2'
    train_image_dir = f"input/{exp_path}/train/{sensor}"  # Cambia esta ruta
    train_mask_dir = f"input/labels/segmentation/train/{sensor}"    # Cambia esta ruta
    val_image_dir = f"input/{exp_path}/val/{sensor}"      # Cambia esta ruta
    val_mask_dir = f"input/labels/segmentation/val/{sensor}"        # Cambia esta ruta

    # Hiperparámetros para el entrenamiento
    hyperparameters = {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'epochs': 1,
        'device': 'cuda',  # Usa 'cuda' para GPU, 'cpu' para CPU
        'num_workers': 4
    }

    # Mapeo de clases (nombre de la clase a ID)
    class_mapping = {
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

    # Dimensiones de las imágenes (ancho, alto)
    image_size = (640, 480)

    # Carpeta de salida para métricas y modelos
    output_dir = f"output/ddbb-s/segmentation/{sensor}_exp{exp_path[:1]}_"

    # Número de clases (incluyendo fondo)
    num_classes = len(class_mapping)

    # Generar el YAML
    output_file = f'data/{exp_path[:1]}_data_config.yaml'  # Nombre del archivo YAML generado
    generate_yaml(output_file, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, num_classes, hyperparameters, class_mapping, image_size, output_dir)
