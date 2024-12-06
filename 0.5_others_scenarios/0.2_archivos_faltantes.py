def determinar_rutas_faltantes(archivo1, archivo2, salida_txt):
    """
    Determina qué rutas están presentes en archivo1 pero faltan en archivo2.

    Args:
        archivo1 (str): Ruta al archivo que contiene todas las rutas.
        archivo2 (str): Ruta al archivo que le faltan rutas.
        salida_txt (str): Ruta al archivo de salida con las rutas faltantes.

    Returns:
        None
    """
    try:
        # Leer rutas de archivo1 en un conjunto
        with open(archivo1, 'r') as f1:
            rutas1 = set(line.strip() for line in f1)

        # Leer rutas de archivo2 en un conjunto
        with open(archivo2, 'r') as f2:
            rutas2 = set(line.strip() for line in f2)

        # Determinar las rutas faltantes
        rutas_faltantes = rutas1 - rutas2

        # Escribir las rutas faltantes en el archivo de salida
        with open(salida_txt, 'w') as salida:
            for ruta in rutas_faltantes:
                salida.write(ruta + '\n')

        print(f"Archivo de salida generado exitosamente: {salida_txt}")
        print(f"Total de rutas faltantes: {len(rutas_faltantes)}")

    except Exception as e:
        print(f"Error: {e}")

# Parámetros
archivo1 = "data/files_1_without_back_without_blur.txt"  # Archivo con todas las rutas
archivo2 = "data/files_3_ddbs-s_with_back_without_blur.txt"  # Archivo con rutas incompletas
salida_txt = "data/rutas_faltantes.txt"  # Archivo donde se guardarán las rutas faltantes

# Llamada a la función
determinar_rutas_faltantes(archivo1, archivo2, salida_txt)
