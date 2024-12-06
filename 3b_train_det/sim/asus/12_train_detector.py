from ultralytics import YOLO

# Define el ancho y alto de las imágenes
img_width = 640  # Cambia este valor por el ancho de tus imágenes
img_height = 480  # Cambia este valor por el alto de tus imágenes

# Cargar el modelo preentrenado YOLOv8
model = YOLO('yolo11n.pt')  # Puedes cambiar a otro modelo preentrenado (yolov8s.pt, etc.)

yaml_file = 'data/1_data_config.yaml'

# Entrenar el modelo usando el archivo .yaml y las dimensiones especificadas
model.train(
    data=yaml_file,      # Ruta al archivo .yaml con la configuración del dataset
    epochs=1,                    # Número de épocas
    batch=128,                     # Tamaño del batch
    imgsz=(img_width, img_height),  # Dimensiones personalizadas de las imágenes
    device=0,                     # GPU específica (0 para la primera GPU, o 'cpu' para CPU)
    workers=4,                    # Número de workers para la carga de datos
    project='output/ddbb-s/detection/',        # Carpeta donde se guardarán los resultados
    name='asus_exp1_' # Nombre del experimento
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
