import os

def contar_archivos_en_nombres(directorio, palabras_clave, escenarios):
    resultados = {palabra: {escenario: 0 for escenario in escenarios} for palabra in palabras_clave}

    for root, _, archivos in os.walk(directorio):
        for archivo in archivos:
            nombre_archivo = archivo.lower()  # Convertir el nombre a minúsculas para una búsqueda insensible a mayúsculas/minúsculas
            for palabra in palabras_clave:
                for escenario in escenarios:
                    if palabra in nombre_archivo and escenario in nombre_archivo:
                        resultados[palabra][escenario] += 1

    return resultados

# Parámetros
directorio = "output/1_without_back_without_blur/"
palabras_clave = ["asus", "davis346", "evk4", "zed2"]
escenarios = ["scn1", "scn2", "scn3", "scn4"]

# Contar archivos
resultados = contar_archivos_en_nombres(directorio, palabras_clave, escenarios)

# Imprimir resultados
for palabra, escenarios_encontrados in resultados.items():
    print(f"\nResultados para '{palabra}':")
    for escenario, cantidad in escenarios_encontrados.items():
        print(f"  - Con '{escenario}': {cantidad} archivo(s)")
