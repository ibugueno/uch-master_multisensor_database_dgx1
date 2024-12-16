def find_missing_line(file1, file2, output_file):
    # Leer las líneas de ambos archivos y eliminar saltos de línea
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines_file1 = set(f1.read().splitlines())
        lines_file2 = set(f2.read().splitlines())
    
    # Encontrar las líneas que están en un archivo pero no en el otro
    missing_in_file1 = lines_file2 - lines_file1
    missing_in_file2 = lines_file1 - lines_file2

    # Guardar los resultados en un archivo
    with open(output_file, 'w') as out:
        if missing_in_file1:
            out.write("Líneas faltantes en el primer archivo:\n")
            out.writelines(line + '\n' for line in sorted(missing_in_file1))
        if missing_in_file2:
            out.write("\nLíneas faltantes en el segundo archivo:\n")
            out.writelines(line + '\n' for line in sorted(missing_in_file2))
    
    print(f"Comparación completa. Resultados guardados en {output_file}")

# Ruta a los archivos
sensor = 'evk4'
model = 'clean'
file1 = f"data/target_{sensor}_list.txt"
file2 = f"data/{sensor}_{model}_aedat_files_list.txt"
output_file = f"data/missing_lines_{sensor}_{model}.txt"

# Ejecutar la función
find_missing_line(file1, file2, output_file)
