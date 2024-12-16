from ultralytics import YOLO
import os
import csv

# Define el ancho y alto de las imágenes
img_width = 1280
img_height = 720

# Obtener el nombre del archivo actual y extraer información
current_file = os.path.basename(__file__)
exp = int(current_file[:1])
sensor = 'evk4'

# Ruta a los pesos preentrenados de YOLO
pretrained_path = f'output/ddbb-s-events/detection/{sensor}_exp{exp}_'
pretrained_weights_path = f'{pretrained_path}/weights/best.pt'

# Cargar el modelo preentrenado YOLO
model = YOLO(pretrained_weights_path)

# Ruta al archivo .yaml de configuración del dataset
yaml_file = f'data/{sensor}_{current_file[:1]}_data_config.yaml'

# Inferencia en el conjunto de validación
results = model.val(
    data=yaml_file,      # Usa la misma configuración del dataset
    imgsz=(img_width, img_height),  # Dimensiones personalizadas de las imágenes
    split='val'                  # Evalúa en la partición de validación definida en el YAML
)

# Extraer métricas desde results.results_dict
metrics = {
    "precision(B)": results.results_dict['metrics/precision(B)'],  # Precisión
    "recall(B)": results.results_dict['metrics/recall(B)'],        # Recall
    "mAP50(B)": results.results_dict['metrics/mAP50(B)'],          # mAP@50
    "mAP50-95(B)": results.results_dict['metrics/mAP50-95(B)']     # mAP promedio (IoU 0.5-0.95)
}
# Guardar las métricas en un archivo CSV
results_csv_path = f'{pretrained_path}/results.csv'
with open(results_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escribir encabezados
    writer.writerow(metrics.keys())
    # Escribir métricas
    writer.writerow(metrics.values())

print(f"Métricas guardadas en: {results_csv_path}")
