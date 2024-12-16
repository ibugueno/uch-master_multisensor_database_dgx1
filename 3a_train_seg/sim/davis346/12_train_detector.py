import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import defaultdict
import random
import time

# Dataset personalizado para segmentación
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_mapping, transforms=None, use_half=False, split_data=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.class_mapping = class_mapping

        # Filtrar y contar máscaras por clase
        self.class_counts = defaultdict(int)
        for mask_file in self.mask_files:
            for class_name, class_id in class_mapping.items():
                if class_name in mask_file:
                    self.class_counts[class_name] += 1
                    break

        # Usar solo una fracción de los datos de forma aleatoria
        if use_half:
            total_files = len(self.image_files)
            selected_indices = random.sample(range(total_files), total_files // split_data)
            self.image_files = [self.image_files[i] for i in selected_indices]
            self.mask_files = [self.mask_files[i] for i in selected_indices]

        # Actualizar conteo de clases después del muestreo
        self.filtered_class_counts = defaultdict(int)
        for mask_file in self.mask_files:
            for class_name, class_id in class_mapping.items():
                if class_name in mask_file:
                    self.filtered_class_counts[class_name] += 1
                    break

        print("Conteo de clases antes del filtrado:")
        for class_name, count in self.class_counts.items():
            print(f"Clase {class_name}: {count} muestras")
        
        print("\nConteo de clases después del filtrado:")
        for class_name, count in self.filtered_class_counts.items():
            print(f"Clase {class_name}: {count} muestras")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.uint8)

        # Normalizar la máscara binaria y asignar clase
        mask = (mask > 127).astype(np.uint8)  # Binarización explícita
        for class_name, class_id in self.class_mapping.items():
            if class_name in os.path.basename(mask_path):
                mask = mask * class_id
                break

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(mask, dtype=torch.long)


def get_transform():
    def transform(img):
        return F.to_tensor(img)
    return transform

def get_model(num_classes):
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Los resultados se guardarán en: {output_dir}")

def train_model(config):
    setup_output_dir(config['output_dir'])
    train_image_dir = config['train']['images']
    train_mask_dir = config['train']['masks']
    batch_size = config['hyperparameters']['batch_size']
    learning_rate = config['hyperparameters']['learning_rate']
    num_epochs = config['hyperparameters']['epochs']
    num_workers = config['hyperparameters']['num_workers']
    device = torch.device(config['hyperparameters']['device'] if torch.cuda.is_available() else 'cpu')

    dataset = SegmentationDataset(train_image_dir, train_mask_dir, config['classes'], transforms=get_transform(), use_half=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = config['num_classes']
    model = get_model(num_classes)
    model.to(device)

    # Calcular pesos para el balance de clases
    total_samples = sum(dataset.filtered_class_counts.values())
    class_weights = torch.tensor(
        [total_samples / dataset.filtered_class_counts.get(cls_name, 1) for cls_name in dataset.class_mapping.keys()],
        dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        progress_bar = tqdm(data_loader, desc=f"Época {epoch + 1}/{num_epochs}")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images)['out']
            loss = torch.nn.functional.cross_entropy(output, masks, weight=class_weights)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (len(progress_bar) + 1)
            elapsed_time = time.time() - start_time
            estimated_time = (elapsed_time / (progress_bar.n + 1)) * (len(progress_bar) - progress_bar.n)

            progress_bar.set_postfix({
                "Pérdida Promedio": f"{avg_loss:.4f}",
                "Tiempo Restante": f"{estimated_time:.2f}s"
            })

        print(f"Época [{epoch + 1}/{num_epochs}] finalizada. Pérdida Total: {epoch_loss:.4f}")

    model_path = os.path.join(config['output_dir'], "deeplabv3_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo entrenado guardado en: {model_path}")

def validate_model(config):
    val_image_dir = config['val']['images']
    val_mask_dir = config['val']['masks']
    output_dir = config['output_dir']
    batch_size = config['hyperparameters']['batch_size']
    num_workers = config['hyperparameters']['num_workers']
    num_classes = config['num_classes']
    device = torch.device(config['hyperparameters']['device'] if torch.cuda.is_available() else 'cpu')

    dataset = SegmentationDataset(val_image_dir, val_mask_dir, config['classes'], transforms=get_transform(), use_half=True, split_data=10)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

            for cls in range(num_classes):
                tp = np.sum((preds == cls) & (masks == cls))
                fp = np.sum((preds == cls) & (masks != cls))
                fn = np.sum((preds != cls) & (masks == cls))

                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

                metrics["IoU"][cls] += iou
                metrics["Precision"][cls] += precision
                metrics["Recall"][cls] += recall
                metrics["F1-Score"][cls] += f1
                metrics["Class Count"][cls] += 1

    for key in ["IoU", "Precision", "Recall", "F1-Score"]:
        metrics[key] = metrics[key] / np.maximum(metrics["Class Count"], 1)

    pixel_accuracy = total_correct / total_pixels
    mean_iou = np.mean(metrics["IoU"])

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
    sensor = 'davis346'
    config_path = f"data/{sensor}_{current_file[:1]}_data_config.yaml"
    config = load_config(config_path)
    train_model(config)
    validate_model(config)
