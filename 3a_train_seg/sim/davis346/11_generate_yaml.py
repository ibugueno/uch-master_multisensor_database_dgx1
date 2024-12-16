import yaml
import os

def generate_yaml(output_file, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, num_classes, hyperparameters, class_mapping, image_size, output_dir):
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Archivo YAML generado: {output_file}")

if __name__ == "__main__":
    exp_path = '1_without_back_clean'
    sensor = 'davis346'
    train_image_dir = f"input/ddbb-s-events/{exp_path}/train/{sensor}"
    train_mask_dir = f"input/labels-events/segmentation/train/{sensor}"
    val_image_dir = f"input/ddbb-s-events/{exp_path}/val/{sensor}"
    val_mask_dir = f"input/labels-events/segmentation/val/{sensor}"

    hyperparameters = {
        'batch_size': 128,
        'learning_rate': 0.0001,
        'epochs': 2,
        'device': 'cuda',
        'num_workers': 4
    }

    class_mapping = {
        'fondo': 0,
        'almohada': 1,
        'arbol': 2,
        'avion': 3,
        'boomerang': 4,
        'caja_amarilla': 5,
        'caja_azul': 6,
        'carro_rojo': 7,
        'clorox': 8,
        'dino': 9,
        'disco': 10,
        'jarron': 11,
        'lysoform': 12,
        'mobil': 13,
        'paleta': 14,
        'pelota': 15,
        'sombrero': 16,
        'tarro': 17,
        'tazon': 18,
        'toalla_roja': 19,
        'zapatilla': 20
    }

    image_size = (346, 260)
    output_dir = f"output/ddbb-s-events/segmentation/{sensor}_exp{exp_path[:1]}_"
    num_classes = len(class_mapping)

    output_file = f'data/{sensor}_{exp_path[:1]}_data_config.yaml'
    generate_yaml(output_file, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, num_classes, hyperparameters, class_mapping, image_size, output_dir)
