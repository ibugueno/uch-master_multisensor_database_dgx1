from ultralytics import YOLO
import os

# Define el ancho y alto de las imágenes
img_width = 346  # Cambia este valor por el ancho de tus imágenes
img_height = 260  # Cambia este valor por el alto de tus imágenes

current_file = os.path.basename(__file__)
exp = int(current_file[:1])
sensor = 'davis346'

# Ruta a los pesos preentrenados de YOLO
pretrained_weights_path = f'output/ddbb-s-events/detection/{sensor}_exp{exp-1}_/weights/best.pt'  # Cambia esta ruta

# Cargar el modelo preentrenado YOLOv8
model = YOLO(pretrained_weights_path)  # Puedes cambiar a otro modelo preentrenado (yolov8s.pt, etc.)

yaml_file = f'data/{sensor}_{current_file[:1]}_data_config.yaml'

# Entrenar el modelo usando el archivo .yaml y las dimensiones especificadas
model.train(
    data=yaml_file,      # Ruta al archivo .yaml con la configuración del dataset
    epochs=10,                    # Número de épocas
    batch=512,                     # Tamaño del batch
    imgsz=(img_width, img_height),  # Dimensiones personalizadas de las imágenes
    device=0,                     # GPU específica (0 para la primera GPU, o 'cpu' para CPU)
    workers=4,                    # Número de workers para la carga de datos
    project='output/ddbb-s-events/detection/',        # Carpeta donde se guardarán los resultados
    name=f'{sensor}_exp{current_file[:1]}_' # Nombre del experimento
)

# Al finalizar el entrenamiento, puedes ver las métricas finales:
metrics = model.metrics
print("Entrenamiento finalizado. Métricas finales:")
print(metrics)

# Evaluar el modelo en el conjunto de test con las dimensiones personalizadas
results = model.val(
    data=yaml_file,      # Usa la misma configuración del dataset
    imgsz=(img_width, img_height), # Dimensiones personalizadas de las imágenes
    split='test'                  # Evalúa en la partición de test definida en el YAML
)

# Imprimir métricas de evaluación
print("Resultados en el conjunto de test:")
print(results)
