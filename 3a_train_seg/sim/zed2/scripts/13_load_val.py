import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from segmentation_models_pytorch import DeepLabV3
import random

# Dataset personalizado para segmentación
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, use_half=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
        # Usar solo la mitad de los datos de forma aleatoria
        if use_half:
            total_files = len(self.image_files)
            #random.seed(42)
            selected_indices = random.sample(range(total_files), total_files // 10)
            self.image_files = [self.image_files[i] for i in selected_indices]
            self.mask_files = [self.mask_files[i] for i in selected_indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.uint8) // 255

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(mask, dtype=torch.long)

def get_transform():
    def transform(img):
        return F.to_tensor(img)
    return transform

def get_model(num_classes):
    # Usar DeepLabV3 con backbone MobileNetV3
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def validate_model(config):
    val_image_dir = config['val']['images']
    val_mask_dir = config['val']['masks']
    output_dir = config['output_dir']
    batch_size = config['hyperparameters']['batch_size']  # Usar el mismo batch_size que en entrenamiento
    num_workers = config['hyperparameters']['num_workers']
    num_classes = config['num_classes']
    device = torch.device(config['hyperparameters']['device'] if torch.cuda.is_available() else 'cpu')

    # Cargar el dataset de validación
    dataset = SegmentationDataset(val_image_dir, val_mask_dir, transforms=get_transform(), use_half=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Cargar el modelo guardado
    model = get_model(num_classes)
    model_path = os.path.join(output_dir, "deeplabv3_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    metrics = {"IoU": np.zeros(num_classes), 
               "Precision": np.zeros(num_classes), 
               "Recall": np.zeros(num_classes), 
               "F1-Score": np.zeros(num_classes),
               "Class Count": np.zeros(num_classes)}
    total_correct = 0
    total_pixels = 0

    progress_bar = tqdm(data_loader, desc="Validación")
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            output = model(images)['out']
            preds = torch.argmax(output, dim=1).cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()

            total_correct += np.sum(preds == masks)
            total_pixels += len(masks)

            # Calcular métricas por clase
            for cls in range(num_classes):
                tp = np.sum((preds == cls) & (masks == cls))
                fp = np.sum((preds == cls) & (masks != cls))
                fn = np.sum((preds != cls) & (masks == cls))

                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

                # Acumular métricas por clase
                metrics["IoU"][cls] += iou
                metrics["Precision"][cls] += precision
                metrics["Recall"][cls] += recall
                metrics["F1-Score"][cls] += f1
                metrics["Class Count"][cls] += 1  # Contador de ejemplos por clase

    # Calcular promedio por clase
    for key in ["IoU", "Precision", "Recall", "F1-Score"]:
        metrics[key] = metrics[key] / np.maximum(metrics["Class Count"], 1)  # Evitar división por 0

    pixel_accuracy = total_correct / total_pixels
    mean_iou = np.mean(metrics["IoU"])

    # Guardar métricas en un archivo de texto
    metrics_path = os.path.join(output_dir, "metrics_validation.txt")
    with open(metrics_path, "w") as file:
        file.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
        file.write(f"Mean IoU: {mean_iou:.4f}\n\n")
        file.write("Metrics per class:\n")
        for cls in range(num_classes):
            file.write(f"Class {cls}:\n")
            file.write(f"  IoU: {metrics['IoU'][cls]:.4f}\n")
            file.write(f"  Precision: {metrics['Precision'][cls]:.4f}\n")
            file.write(f"  Recall: {metrics['Recall'][cls]:.4f}\n")
            file.write(f"  F1-Score: {metrics['F1-Score'][cls]:.4f}\n\n")

    print(f"Validación completa. Resultados guardados en: {metrics_path}")


if __name__ == "__main__":
    current_file = os.path.basename(__file__)
    config_path = f"data/{current_file[:1]}_data_config.yaml"
    config = load_config(config_path)
    validate_model(config)
