from functions import procesar_archivos_abf

# Definir las rutas de la carpeta fuente y la carpeta de destino
carpeta_abf_source = './data/source'
carpeta_segments_destino = './data/segments'

# Llamar a la función para procesar los archivos .abf
procesar_archivos_abf(carpeta_abf_source, carpeta_segments_destino)