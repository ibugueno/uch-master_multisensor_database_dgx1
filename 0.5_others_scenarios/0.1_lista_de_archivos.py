import os

def generar_lista_archivos_relativos(directorio, salida_txt):
    """
    Genera un archivo .txt con las rutas relativas de todos los archivos contenidos de forma recursiva en un directorio.

    Args:
        directorio (str): Ruta al directorio raíz.
        salida_txt (str): Nombre del archivo de salida.

    Returns:
        None
    """
    try:
        with open(salida_txt, 'w') as salida:
            for root, _, archivos in os.walk(directorio):
                for archivo in archivos:
                    ruta_archivo = os.path.relpath(os.path.join(root, archivo), start=directorio)
                    salida.write(ruta_archivo + '\n')
        print(f"Archivo generado exitosamente: {salida_txt}")
    except Exception as e:
        print(f"Error al generar el archivo: {e}")

# Parámetros
name = '3_ddbs-s_with_back_without_blur'
directorio = f"output/{name}"      # Cambia esto a tu directorio objetivo
salida_txt = f"data/files_{name}.txt"  # Nombre del archivo de salida

# Llamada a la función
generar_lista_archivos_relativos(directorio, salida_txt)
