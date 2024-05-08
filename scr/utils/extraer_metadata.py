import pyabf
import pandas as pd

def extraer_metadatos(abf_path):
    abf = pyabf.ABF(abf_path)
    metadatos = {}
    for attr in dir(abf):
        if not callable(getattr(abf, attr)) and not attr.startswith("_"):
            valor = getattr(abf, attr)
            # Intentar convertir cualquier objeto que no sea de tipo básico a string
            if not isinstance(valor, (str, int, float, bool)):
                try:
                    valor = str(valor)
                except:
                    valor = "Error al convertir valor"
            metadatos[attr] = valor
    return metadatos

def procesar_archivos(archivos):
    # Inicializar una lista para todos los metadatos
    todos_metadatos = []

    for archivo in archivos:
        metadatos = extraer_metadatos(archivo)
        for propiedad, valor in metadatos.items():
            # Para cada propiedad, crear una fila con el nombre del archivo, la propiedad y su valor
            todos_metadatos.append({"Archivo": archivo.split('/')[-1], "Propiedad": propiedad, "Valor": valor})
    
    return pd.DataFrame(todos_metadatos)

# Lista de archivos ABF para analizar
archivos_abf = [
    'data/source/15o14000.abf',
    'data/source/15o14006.abf',
    'data/source/15o14024.abf',
    'data/source/15o14031.abf',
    'data/source/17613000.abf',
    'data/source/17613006.abf',
    'data/source/17613025.abf',
    'data/source/17613029.abf',
    'data/source/17620000.abf',
    'data/source/17620007.abf',
    'data/source/17620028.abf',
    'data/source/17620042.abf']

# Procesar los archivos y extraer metadatos
df_metadatos = procesar_archivos(archivos_abf)

# Guardar en Excel
df_metadatos.to_excel("metadatos_todos.xlsx", index=False)