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
import time
from segmentation_models_pytorch import DeepLabV3

# Dataset personalizado para segmentación
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, use_half=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
        # Usar solo la mitad de los datos
        if use_half:
            half_size = len(self.image_files) // 2
            self.image_files = self.image_files[:half_size]
            self.mask_files = self.mask_files[:half_size]

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

    dataset = SegmentationDataset(train_image_dir, train_mask_dir, transforms=get_transform(), use_half=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = config['num_classes']
    model = get_model(num_classes)
    model.to(device)

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
            loss = torch.nn.functional.cross_entropy(output, masks)
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

def evaluate_model(config):
    val_image_dir = config['val']['images']
    val_mask_dir = config['val']['masks']
    output_dir = config['output_dir']
    dataset = SegmentationDataset(val_image_dir, val_mask_dir, transforms=get_transform())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['hyperparameters']['num_workers'])
    num_classes = config['num_classes']
    model = get_model(num_classes)
    model_path = os.path.join(output_dir, "deeplabv3_model.pth")
    model.load_state_dict(torch.load(model_path))
    device = torch.device(config['hyperparameters']['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    metrics = {"IoU": [], "Precision": [], "Recall": [], "F1-Score": []}
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

                metrics["Precision"].append(precision)
                metrics["Recall"].append(recall)
                metrics["F1-Score"].append(f1)
                metrics["IoU"].append(iou)

            avg_iou = np.mean(metrics["IoU"])
            progress_bar.set_postfix({"IoU Promedio": f"{avg_iou:.4f}"})

    pixel_accuracy = total_correct / total_pixels

    # Guardar métricas en un archivo de texto
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as file:
        file.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
        file.write(f"Mean IoU: {avg_iou:.4f}\n")


if __name__ == "__main__":
    current_file = os.path.basename(__file__)
    config_path = f"data/{current_file[:1]}_data_config.yaml"
    config = load_config(config_path)
    train_model(config)
    evaluate_model(config)


