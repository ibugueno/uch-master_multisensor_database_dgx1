def count_lines_in_common(file1, file2):
    """
    Cuenta cuántas líneas del primer archivo están presentes en el segundo archivo,
    eliminando prefijos específicos antes de la comparación.

    Args:
        file1 (str): Ruta al primer archivo.
        file2 (str): Ruta al segundo archivo.

    Returns:
        int: Número de líneas del primer archivo presentes en el segundo archivo.
    """
    # Prefijos a eliminar
    prefix1 = "output/ddbb-s-events/1_without_back_clean/"
    prefix2 = "output/ddbb-s-events/2_without_back_noisy/"

    # Leer y procesar las líneas del primer archivo
    with open(file1, 'r') as f1:
        lines1 = set(line.strip().replace(prefix1, "") for line in f1)

    # Leer y procesar las líneas del segundo archivo
    with open(file2, 'r') as f2:
        lines2 = set(line.strip().replace(prefix2, "") for line in f2)

    # Calcular cuántas líneas están en ambos conjuntos
    common_lines = lines1.intersection(lines2)

    return len(common_lines)


# Rutas a los archivos
file1_path = 'data/output_ddbb-s-events_1_without_back_clean_evk4_files.txt'
file2_path = 'data/output_ddbb-s-events_2_without_back_noisy_evk4_files.txt'

# Contar líneas en común
common_count = count_lines_in_common(file1_path, file2_path)

print(f"El número de líneas del primer archivo presentes en el segundo archivo es: {common_count}")
