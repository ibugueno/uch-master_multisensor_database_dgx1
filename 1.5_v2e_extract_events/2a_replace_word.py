def replace_word_in_file(input_file, output_file, old_word, new_word):
    """
    Reemplaza una palabra en cada línea de un archivo y guarda el resultado en un nuevo archivo.

    Args:
        input_file (str): Ruta del archivo de entrada.
        output_file (str): Ruta del archivo de salida.
        old_word (str): Palabra que se reemplazará.
        new_word (str): Palabra de reemplazo.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Reemplazar la palabra y escribir en el archivo de salida
                outfile.write(line.replace(old_word, new_word))
        print(f"Archivo procesado exitosamente. Nuevo archivo guardado en: {output_file}")
    except FileNotFoundError:
        print(f"El archivo '{input_file}' no se encontró.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso
input_file_path = "data/target_davis346_list.txt"  # Ruta al archivo de entrada
output_file_path = "data/target_evk4_list.txt"  # Ruta al archivo de salida
replace_word_in_file(input_file_path, output_file_path, "davis346", "evk4")
