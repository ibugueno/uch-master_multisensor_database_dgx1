import os
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import yaml
import torchvision.models as models
from tqdm import tqdm
import contextlib
import io
import logging
import random  # Importar random para selección aleatoria

# -------------------------------------
# Configurar el logging de YOLO para silenciar las salidas
# -------------------------------------
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# -------------------------------------
# Selección aleatoria de imágenes
# -------------------------------------
def select_random_subset(files, fraction=0.5, seed=42):
    """Selecciona aleatoriamente un subconjunto de archivos."""
    random.seed(seed)  # Fija la semilla para reproducibilidad
    subset_size = int(len(files) * fraction)  # Calcula el tamaño del subconjunto
    return random.sample(files, subset_size)


# -------------------------------------
# Modelo de Regresión de Pose
# -------------------------------------
class PoseRegressor(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseRegressor, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce a [batch_size, 512, 1, 1]

        # Reemplazar la capa fc de ResNet para que acepte [batch_size, 512]
        self.cnn.fc = nn.Linear(512, 5 + num_classes)  # Salida para pose (5) + clasificación (num_classes)

    def forward(self, image):
        x = self.cnn.conv1(image)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)  # Reduce a [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)  # Convierte a [batch_size, 512]
        x = self.cnn.fc(x)  # Salida final [batch_size, 5 + num_classes]

        pose = x[:, :5]           # Primeros 5 valores para pose
        class_logits = x[:, 5:]   # Resto para clasificación
        return pose, class_logits


# -------------------------------------
# Transformaciones para la Imagen
# -------------------------------------
def preprocess_image(image, bbox, input_size=(224, 224)):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize(input_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(cropped_image).unsqueeze(0)  # Añadir dimensión batch

# -------------------------------------
# Métricas de Error
# -------------------------------------
def calculate_metrics(predictions, ground_truths, xy_predictions, xy_ground_truths):
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    xy_predictions = np.array(xy_predictions)
    xy_ground_truths = np.array(xy_ground_truths)

    # Métricas para pose
    mse = mean_squared_error(ground_truths, predictions)
    mae = mean_absolute_error(ground_truths, predictions)
    z_mae = mean_absolute_error(ground_truths[:, 0], predictions[:, 0])  # z_distance
    q_mae = mean_absolute_error(ground_truths[:, 1:], predictions[:, 1:])  # q1, q2, q3, q4

    # Métricas para coordenadas x e y
    x_mae = mean_absolute_error(xy_ground_truths[:, 0], xy_predictions[:, 0])
    y_mae = mean_absolute_error(xy_ground_truths[:, 1], xy_predictions[:, 1])

    return mse, mae, z_mae, q_mae, x_mae, y_mae

# -------------------------------------
# Pipeline Completo
# -------------------------------------
def main():

    # Obtener el nombre del archivo actual y extraer información
    current_file = os.path.basename(__file__)
    exp = int(current_file[:1])

    # Configuraciones
    sensor = 'asus'
    img_width = 640
    img_height = 480

    # Pesos preentrenados
    yolo_weights_path = f'../output/ddbb-s-frames/detection/{sensor}_exp1_/weights/best.pt'
    pose_weights_path = f'../output/ddbb-s-frames/pose6d-fixed/{sensor}_1/pose_model_epoch_1.pth'

    # YAML del dataset
    yaml_file = f'data/{sensor}_{exp}_config.yaml'

    # Leer configuraciones del YAML
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    val_images_path = config['val']['images']
    val_labels_path = config['val']['labels']

    # Cargar YOLO
    yolo_model = YOLO(yolo_weights_path)

    # Cargar el modelo de regresión de pose
    num_classes = 20  # Cambiar según tu dataset
    pose_model = PoseRegressor(num_classes=num_classes)
    pose_model.load_state_dict(torch.load(pose_weights_path, map_location='cuda'))
    pose_model.eval()

    # Dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_model.to(device)

    # Evaluar en el conjunto de validación
    results_dir = f'../output/ddbb-s-frames/pose6d-fixed/{sensor}_{exp}/val'
    os.makedirs(results_dir, exist_ok=True)

    # Archivo consolidado para guardar métricas finales
    metrics_file = os.path.join(results_dir, "metrics_summary.txt")

    # Listas para métricas
    predictions = []
    ground_truths = []
    xy_predictions = []
    xy_ground_truths = []

    # Iterar sobre las imágenes de validación con tqdm
    image_files = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.png'))]

    # Seleccionar aleatoriamente la mitad de las imágenes
    selected_files = select_random_subset(image_files, fraction=0.1)


    for image_name in tqdm(selected_files, desc="Procesando imagenes de validacion", leave=False):

        # Cargar imagen y etiqueta
        image_path = os.path.join(val_images_path, image_name)
        label_path = os.path.join(val_labels_path, image_name.replace('.jpg', '.txt'))
        image = Image.open(image_path).convert('RGB')

        # Imprimir dimensiones de la imagen
        #print(f"Dimensiones de la imagen: {image.size}")

        # Leer etiqueta
        with open(label_path, 'r') as f:
            label = f.readline().strip().split(',')
        x_gt, y_gt, z_gt, q1_gt, q2_gt, q3_gt, q4_gt = map(float, label[1:8])

        # Suprimir salida estándar de YOLO.predict
        with contextlib.redirect_stdout(io.StringIO()):
            detections = yolo_model.predict(image_path, imgsz=(img_width, img_height), device=device)[0]



        for detection in detections.boxes:
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0].tolist())

            # Preprocesar la imagen recortada
            cropped_tensor = preprocess_image(image, (x_min, y_min, x_max, y_max)).to(device)

            # Estimar pose
            pose, _ = pose_model(cropped_tensor)

            # Extraer valores individuales de la pose
            z_pred = pose[0, 0].cpu().item()  # Primer valor (z)
            q1_pred = pose[0, 1].cpu().item()  # Segundo valor (q1)
            q2_pred = pose[0, 2].cpu().item()  # Tercer valor (q2)
            q3_pred = pose[0, 3].cpu().item()  # Cuarto valor (q3)
            q4_pred = pose[0, 4].cpu().item()  # Quinto valor (q4)

            # Calcular \(x_{pred}\) y \(y_{pred}\) normalizados
            x_pred = ((x_min + x_max) / 2) / img_width
            y_pred = ((y_min + y_max) / 2) / img_height

            # Guardar predicciones y ground truths para métricas
            predictions.append([z_pred, q1_pred, q2_pred, q3_pred, q4_pred])
            ground_truths.append([z_gt, q1_gt, q2_gt, q3_gt, q4_gt])
            xy_predictions.append([x_pred, y_pred])
            xy_ground_truths.append([x_gt, y_gt])

    # Calcular métricas
    mse, mae, z_mae, q_mae, x_mae, y_mae = calculate_metrics(predictions, ground_truths, xy_predictions, xy_ground_truths)

    # Guardar métricas finales
    with open(metrics_file, 'w') as f:
        f.write("\n--- Métricas Finales ---\n")
        f.write(f"MSE Total: {mse:.4f}\n")
        f.write(f"MAE Total: {mae:.4f}\n")
        f.write(f"MAE Z-Distance: {z_mae:.4f}\n")
        f.write(f"MAE Quaternions: {q_mae:.4f}\n")
        f.write(f"MAE X-Coordinate: {x_mae:.4f}\n")
        f.write(f"MAE Y-Coordinate: {y_mae:.4f}\n")

    # Imprimir métricas
    print("\n--- Métricas Finales ---")
    print(f"MSE Total: {mse:.4f}")
    print(f"MAE Total: {mae:.4f}")
    print(f"MAE Z-Distance: {z_mae:.4f}")
    print(f"MAE Quaternions: {q_mae:.4f}")
    print(f"MAE X-Coordinate: {x_mae:.4f}")
    print(f"MAE Y-Coordinate: {y_mae:.4f}")

if __name__ == "__main__":
    main()
