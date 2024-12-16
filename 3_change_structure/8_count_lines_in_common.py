def generate_missing_lines(file1, file2, output_txt):
    """
    Genera un archivo .txt con las líneas del primer archivo que no están presentes en el segundo archivo.

    Args:
        file1 (str): Ruta al primer archivo.
        file2 (str): Ruta al segundo archivo.
        output_txt (str): Ruta al archivo .txt de salida.
    """
    # Leer y procesar las líneas del primer archivo
    with open(file1, 'r') as f1:
        lines1 = set(line.strip() for line in f1)

    # Leer y procesar las líneas del segundo archivo
    with open(file2, 'r') as f2:
        lines2 = set(line.strip() for line in f2)

    # Calcular las líneas que están en file1 pero no en file2
    missing_lines = lines1 - lines2

    # Escribir las líneas faltantes en el archivo de salida
    with open(output_txt, 'w') as out_f:
        out_f.writelines(f"{line}\n" for line in missing_lines)

    print(f"Se generó un archivo con {len(missing_lines)} líneas que no están en el segundo archivo.")
    print(f"Total de líneas en el primer archivo: {len(lines1)}")
    print(f"Total de líneas en el segundo archivo: {len(lines2)}")
    print(f"Líneas faltantes: {len(missing_lines)}")

sensor = 'davis346'

# Rutas a los archivos
file1_path = f'data/{sensor}_output_ddbb-s-events_3_with_back_clean__files.txt'
file2_path = f'data/{sensor}_output_labels-events_segmentation__files.txt'
output_txt_path = f'data/{sensor}_missing_files.txt'  # Archivo de salida con las líneas faltantes

# Generar el archivo con las líneas faltantes
generate_missing_lines(file1_path, file2_path, output_txt_path)
