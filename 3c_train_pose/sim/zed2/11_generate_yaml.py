import yaml
import os

def generate_yaml(output_file, train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir, num_classes, hyperparameters, class_mapping, output_dir):
    config = {
        'train': {
            'images': train_image_dir,
            'labels': train_label_dir,
        },
        'val': {
            'images': val_image_dir,
            'labels': val_label_dir,
        },
        'test': {
            'images': test_image_dir,
            'labels': test_label_dir,
        },
        'num_classes': num_classes,
        'hyperparameters': hyperparameters,
        'classes': class_mapping,
        'output_dir': output_dir
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Archivo YAML generado: {output_file}")

if __name__ == "__main__":
    # Configuración para el sensor y experimento
    exp_path = '1_without_back_without_blur'
    sensor = 'zed2'
    root_data = '../input/ddbb-s-frames/data'
    root_labels = '../input/ddbb-s-frames/labels'
    path_task = 'pose6d-fixed'

    train_image_dir = f"{root_data}/{exp_path}/train/{sensor}"
    train_label_dir = f"{root_labels}/{path_task}/train/{sensor}"
    
    val_image_dir = f"{root_data}/{exp_path}/val/{sensor}"
    val_label_dir = f"{root_labels}/{path_task}/val/{sensor}"

    test_image_dir = f"{root_data}/{exp_path}/test/{sensor}"
    test_label_dir = f"{root_labels}/{path_task}/test/{sensor}"

    hyperparameters = {
        'batch_size': 128,
        'learning_rate': 0.0001,
        'epochs': 1,
        'device': 'cuda',
        'num_workers': 4
    }

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

    output_dir = f"../output/ddbb-s-frames/{path_task}/{sensor}_{exp_path[:1]}"
    num_classes = len(class_mapping)
    output_file = f"data/{sensor}_{exp_path[:1]}_config.yaml"

    generate_yaml(output_file, train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir, num_classes, hyperparameters, class_mapping, output_dir)
