import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

class PoseDataset(Dataset):
    def __init__(self, image_dir, label_dir, partition):
        """
        Dataset para la estimación de pose 6D con imágenes recortadas.
        :param image_dir: Directorio de las imágenes.
        :param label_dir: Directorio de las etiquetas.
        :param partition: 'train', 'val', o 'test'.
        """
        self.image_dir = os.path.join(image_dir, partition)
        self.label_dir = os.path.join(label_dir, partition)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])

        # Transformaciones para normalización
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Leer imagen
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size

        # Leer etiqueta correspondiente
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            label = f.readline().strip().split(',')
        
        # Extraer datos del label
        class_id = int(label[0])  # ID de la clase
        x_cen, y_cen, width, height, z_distance = map(float, label[1:6])
        q1, q2, q3, q4 = map(float, label[6:10])

        # Calcular bounding box en píxeles
        x_min = int((x_cen - width / 2) * img_width)
        x_max = int((x_cen + width / 2) * img_width)
        y_min = int((y_cen - height / 2) * img_height)
        y_max = int((y_cen + height / 2) * img_height)

        # Asegurar que los valores estén dentro del rango válido
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width, x_max), min(img_height, y_max)

        # Recortar la imagen
        image_cropped = image.crop((x_min, y_min, x_max, y_max))

        # Aplicar transformaciones
        image_cropped = self.transforms(image_cropped)

        # Construir la pose (6D)
        pose = torch.tensor([z_distance, q1, q2, q3, q4], dtype=torch.float32)

        return image_cropped, pose, class_id

class PoseRegressor(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseRegressor, self).__init__()
        
        # Usar ResNet18 como extractor de características
        self.cnn = models.resnet18(pretrained=True)
        
        # Reemplazar la última capa para estimación de pose y clasificación
        self.cnn.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Ajustar tamaño espacial a 1x1
            nn.Flatten(),                 # Aplanar para capas densas
            nn.Linear(512, 5 + num_classes)  # Pose (5) + Clasificación (num_classes)
        )

    def forward(self, image):
        output = self.cnn(image)
        pose = output[:, :5]       # Los primeros 5 valores son la pose (z, q1, q2, q3, q4)
        class_logits = output[:, 5:]  # Los restantes son logits de clasificación
        return pose, class_logits

def train_model(model, train_loader, optimizer, pose_criterion, class_criterion, epochs=20, device='cuda'):
    model.train()
    for epoch in range(epochs):
        running_pose_loss = 0.0
        running_class_loss = 0.0
        
        for images, poses, class_ids in train_loader:
            images, poses, class_ids = images.to(device), poses.to(device), class_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_pose, pred_class_logits = model(images)
            
            # Calcular pérdidas
            pose_loss = pose_criterion(pred_pose, poses)
            class_loss = class_criterion(pred_class_logits, class_ids)
            loss = pose_loss + class_loss
            
            # Backpropagation y optimización
            loss.backward()
            optimizer.step()
            
            running_pose_loss += pose_loss.item()
            running_class_loss += class_loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Pose Loss: {running_pose_loss/len(train_loader)}, Class Loss: {running_class_loss/len(train_loader)}")

def evaluate_model(model, test_loader, pose_criterion, class_criterion, device='cuda'):
    model.eval()
    total_pose_loss = 0.0
    total_class_loss = 0.0
    
    with torch.no_grad():
        for images, poses, class_ids in test_loader:
            images, poses, class_ids = images.to(device), poses.to(device), class_ids.to(device)
            
            # Forward pass
            pred_pose, pred_class_logits = model(images)
            
            # Calcular pérdidas
            pose_loss = pose_criterion(pred_pose, poses)
            class_loss = class_criterion(pred_class_logits, class_ids)
            
            total_pose_loss += pose_loss.item()
            total_class_loss += class_loss.item()
    
    print(f"Pose Loss: {total_pose_loss/len(test_loader)}, Class Loss: {total_class_loss/len(test_loader)}")


if __name__ == "__main__":
    # Rutas de las imágenes y etiquetas
    image_dir = "/path/to/images"
    label_dir = "/path/to/labels"

    # Crear datasets
    train_dataset = PoseDataset(image_dir, label_dir, "train")
    val_dataset = PoseDataset(image_dir, label_dir, "val")
    test_dataset = PoseDataset(image_dir, label_dir, "test")
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Configurar modelo, pérdidas y optimizador
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseRegressor(num_classes=20).to(device)
    pose_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenar el modelo
    train_model(model, train_loader, optimizer, pose_criterion, class_criterion, epochs=20, device=device)
    
    # Evaluar el modelo
    evaluate_model(model, test_loader, pose_criterion, class_criterion, device=device)
