def transformar_rutas(archivo_original, archivo_corregido):
    """
    Transforma las rutas en el archivo original eliminando los prefijos y corrigiendo la estructura completa.

    Args:
        archivo_original (str): Ruta al archivo con las rutas originales.
        archivo_corregido (str): Ruta al archivo donde se guardarán las rutas corregidas.

    Returns:
        None
    """
    try:
        with open(archivo_original, 'r') as entrada, open(archivo_corregido, 'w') as salida:
            for linea in entrada:
                # Eliminar prefijos como train/, val/, test/
                ruta = linea.strip().split('/', 1)[-1]

                # Dividir la ruta en partes por '_'
                partes = ruta.split('_')

                # Reconstruir la nueva estructura basada en las partes
                if ('lum' in partes[7]):
                    nueva_ruta = (
                        f"{partes[0]}/{partes[1]}_{partes[2]}_{partes[3]}/scene_{partes[5]}/"
                        f"{partes[6]}/{partes[7]}/{partes[8]}_{partes[9]}_{partes[10]}_{partes[11]}/image_{partes[-1]}"
                    )
                else:
                    nueva_ruta = (
                        f"{partes[0]}/{partes[1]}_{partes[2]}_{partes[3]}/scene_{partes[5]}/"
                        f"{partes[6]}_{partes[7]}/{partes[8]}/{partes[9]}_{partes[10]}_{partes[11]}_{partes[12]}/image_{partes[-1]}"
                    )

                #'asus/asus_scn0_lum9/scene_0/almohada/lum9/orientation_-10_72_-129'

                # Escribir la ruta corregida al archivo de salida
                salida.write(nueva_ruta + '\n')

        print(f"Archivo corregido generado exitosamente: {archivo_corregido}")
    except Exception as e:
        print(f"Error: {e}")

# Parámetros
archivo_original = "data/rutas_faltantes.txt"  # Archivo con rutas originales
archivo_corregido = "data/rutas_corregidas.txt"  # Archivo para guardar rutas corregidas

# Llamada a la función
transformar_rutas(archivo_original, archivo_corregido)
