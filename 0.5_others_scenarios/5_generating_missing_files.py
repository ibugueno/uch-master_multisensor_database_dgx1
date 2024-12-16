import os
import cv2

def seleccionar_fondo_para_archivo(ruta_archivo):
    """
    Selecciona el fondo adecuado basado en la ruta del archivo.
    """
    if "evk4" in ruta_archivo:
        return 'back/back_1_evk4.jpg'
    elif "davis346" in ruta_archivo:
        return 'back/back_1_davis346.jpg'
    else:
        raise ValueError(f"No se reconoce un fondo para el archivo: {ruta_archivo}")

def replicar_estructura_y_guardar(base_salida, relative_path, contenido, extension='.jpg'):
    """
    Guarda contenido en `base_salida` asegurando que los directorios necesarios existen,
    replicando la estructura relativa.
    """
    if extension:
        relative_path = os.path.splitext(relative_path)[0] + extension  # Cambiar la extensión si es necesario
    salida_path = os.path.join(base_salida, relative_path)
    os.makedirs(os.path.dirname(salida_path), exist_ok=True)  # Crear directorios si no existen
    cv2.imwrite(salida_path, contenido)
    return salida_path

def procesar_missing_files(missing_files_path, output_dir):
    """
    Lee el archivo missing_files.txt y genera imágenes de fondo en las rutas especificadas.
    """
    with open(missing_files_path, "r") as file:
        missing_files = [line.strip() for line in file.readlines()]

    for relative_path in missing_files:
        try:
            # Seleccionar el fondo según el archivo
            fondo_path = seleccionar_fondo_para_archivo(relative_path)
            fondo = cv2.imread(fondo_path, cv2.IMREAD_COLOR)
            if fondo is None:
                print(f"[ERROR] No se pudo cargar el fondo: {fondo_path}")
                continue
            
            # Generar la ruta de salida
            ruta_salida = replicar_estructura_y_guardar(output_dir, relative_path, fondo)
            print(f"[INFO] Fondo generado: {ruta_salida}")
        except ValueError as e:
            print(f"[WARNING] {e}")

if __name__ == "__main__":
    # Parámetros
    missing_files_path = "missing_files.txt"  # Archivo que contiene las rutas relativas de los archivos faltantes
    output_dir = "input/frames_all_with_back_without_blur/3_ddbs-s_with_back_without_blur/"  # Directorio de salida para guardar las imágenes de fondo

    # Procesar el archivo missing_files.txt
    procesar_missing_files(missing_files_path, output_dir)
