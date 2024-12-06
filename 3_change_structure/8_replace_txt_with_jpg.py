def replace_txt_with_jpg(input_txt, output_txt):
    """
    Lee un archivo .txt que contiene rutas relativas a archivos .txt,
    reemplaza la extensión .txt por .jpg y guarda el resultado en otro archivo.

    Args:
        input_txt (str): Ruta al archivo .txt con las rutas originales.
        output_txt (str): Ruta al archivo .txt donde se guardarán las rutas modificadas.
    """
    with open(input_txt, "r") as infile, open(output_txt, "w") as outfile:
        for line in infile:
            original_path = line.strip()  # Quita espacios y saltos de línea
            if original_path.endswith(".txt"):
                modified_path = original_path[:-4] + ".jpg"  # Reemplaza .txt por .jpg
                outfile.write(modified_path + "\n")
            else:
                # Si la línea no termina en .txt, escribe la línea sin cambios
                outfile.write(original_path + "\n")

    print(f"Las rutas modificadas se han guardado en '{output_txt}'")

# Especificar rutas
input_txt = "data/labels_pose6d_extra_files.txt"  # Archivo de entrada con rutas a .txt
output_txt = "data/images_extra_files.txt"  # Archivo de salida con rutas modificadas

# Ejecutar la función
replace_txt_with_jpg(input_txt, output_txt)
