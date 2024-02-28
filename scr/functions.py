import pyabf
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from scipy.signal import find_peaks
import os
import pandas as pd
import json
from IPython.display import display

def determinar_modo_grabacion(abf):
    """
    Determina el modo de grabación de un archivo ABF.

    :param abf: Objeto ABF cargado.
    :return: String que describe el modo de grabación.
    """

    if abf.sweepCount > 1:
        # Si hay más de un episodio, es episódico
        # Verificar si todos los episodios tienen la misma longitud
        longitudes_episodios = [abf.sweepLengthInPoints(i) for i in range(abf.sweepCount)]
        if len(set(longitudes_episodios)) == 1:
            return "Episódico Fijo: Datos recopilados en episodios de longitud fija."
        else:
            return "Episódico Variable: Datos recopilados en episodios, pero con longitudes variables."
    else:
        # Si solo hay un "episodio", es gap-free
        return "Gap-Free: Un flujo continuo de datos."

def extraer_metadatos(abf):
    """
    Extrae metadatos de un objeto ABF ya cargado.

    :param abf: Objeto ABF cargado.
    :return: Diccionario con metadatos del objeto ABF.
    """

    metadatos = {
        "nombre_archivo": abf.abfID,
        "frecuencia_muestreo": abf.dataRate,
        "numero_canales": abf.channelCount,
        "numero_episodios": abf.sweepCount,
        "tiempo_total": abf.sweepCount / abf.dataRate,
        "fecha_grabacion": abf.abfDateTimeString,
        "comentarios": abf.abfFileComment,
        "unidad_tiempo": "segundos",
        "unidades_canal": {},
        "etiquetas_canal": {},
        "dimensiones_datos": abf.data.shape
    }

    for i in range(abf.channelCount):
        canal = abf.setSweep(0, channel=i)
        metadatos["unidades_canal"][i] = abf.sweepUnitsY
        metadatos["etiquetas_canal"][i] = abf.sweepLabelY

    return metadatos

def seleccionar_datos(abf, inicio=0, fin=None, episodios='todos', canales='todos'):
    """
    Selecciona segmentos específicos de datos en un archivo ABF.

    :param abf: Objeto ABF cargado.
    :param inicio: Inicio del segmento de tiempo para la selección de datos (en segundos).
    :param fin: Fin del segmento de tiempo para la selección de datos (en segundos).
    :param episodios: Episodios específicos a leer. Puede ser un número, una lista o 'todos'.
    :param canales: Canales específicos a leer. Puede ser un número, una lista o 'todos'.
    :return: Tupla con el tiempo y los datos seleccionados.
    """

    datos_seleccionados = {}
    frecuencia_muestreo = abf.dataRate  # Frecuencia de muestreo en Hz

    # Convertir segundos a puntos de muestreo
    punto_inicio = int(inicio * frecuencia_muestreo)
    punto_fin = int(fin * frecuencia_muestreo) if fin is not None else None

    episodios_a_leer = range(abf.sweepCount) if episodios == 'todos' else episodios
    canales_a_leer = range(abf.channelCount) if canales == 'todos' else canales

    for i in episodios_a_leer:
        abf.setSweep(sweepNumber=i)
        tiempo = abf.sweepX[punto_inicio:punto_fin]  # Tiempo en segundos
        for canal in canales_a_leer:
            abf.setSweep(sweepNumber=i, channel=canal)
            datos_seleccionados[(i, canal)] = abf.sweepY[punto_inicio:punto_fin]

    return tiempo, datos_seleccionados

def graficar_canales_interactivo(tiempo, datos, metadatos, titulo="Datos ABF por Canal"):
    """
    Grafica los datos de cada canal de un archivo ABF de manera interactiva,
    utilizando los metadatos para las unidades.
    """
    
    def actualizar_grafica(episodio=None, canal=None):
        # Limpia la figura actual
        plt.clf()
        
        # Decide qué datos graficar
        if episodio is not None and canal is not None:
            clave = (episodio, canal)
            valores = datos[clave]
            unidad = metadatos["unidades_canal"].get(canal, "Unidad Desconocida")
            plt.plot(tiempo, valores)
            plt.title(f"{titulo} - Episodio {episodio}, Canal {canal}")
            plt.ylabel(unidad)
        else:
            # Graficar todos los canales juntos si no se especifica
            for clave, valores in datos.items():
                episodio, canal = clave
                unidad = metadatos["unidades_canal"].get(canal, "Unidad Desconocida")
                plt.plot(tiempo, valores, label=f"Episodio {episodio}, Canal {canal} ({unidad})")
            plt.title(f"{titulo} - Todos los Canales")
            plt.legend()
        
        plt.xlabel("Tiempo (s)")
        plt.show()
    
    # Widget para elegir episodio y canal
    episodios = list({clave[0] for clave in datos.keys()})
    canales = list({clave[1] for clave in datos.keys()})
    episodio_widget = widgets.Dropdown(options=[None] + episodios, description='Episodio:')
    canal_widget = widgets.Dropdown(options=[None] + canales, description='Canal:')
    
    def actualizar_canal(*args):
        episodio = episodio_widget.value
        canal = canal_widget.value
        actualizar_grafica(episodio, canal)
    
    episodio_widget.observe(actualizar_canal, 'value')
    canal_widget.observe(actualizar_canal, 'value')
    
    display(episodio_widget, canal_widget)

    # Inicialmente, mostrar todos los canales juntos
    actualizar_grafica()

def extraer_segmentos_entre_impulsos(abf, canal, voltaje_umbral=5, distancia_minima_entre_impulsos=1000):
    """
    Extrae y devuelve segmentos de una señal que están entre impulsos definidos por un voltaje umbral,
    para un canal específico de una señal ABF multi-canal.

    :param abf: Objeto ABF cargado, por ejemplo, usando pyABF.
    :param canal: Índice del canal (basado en cero) a procesar.
    :param voltaje_umbral: Umbral de voltaje para detectar impulsos.
    :param distancia_minima_entre_impulsos: Distancia mínima en índices entre picos consecutivos.
    :return: Lista de arrays de NumPy, cada uno representando un segmento de la señal entre impulsos para el canal dado.
    """
    
    segmentos = []

    # Iterar a través de cada episodio (sweep) en el archivo ABF
    for i in range(abf.sweepCount):
        # Establecer el sweep actual y el canal
        abf.setSweep(i, channel=canal)
        
        # Extraer la señal para el episodio y canal actuales
        senal = abf.sweepY
        
        # Detectar los picos (impulsos) basándose en el umbral de voltaje
        picos, _ = find_peaks(senal, height=voltaje_umbral, distance=distancia_minima_entre_impulsos)
        
        # Extraer segmentos entre los picos
        inicio_segmento = 0
        for pico in picos:
            segmentos.append(senal[inicio_segmento:pico])  # Asumiendo que el pico es el inicio del impulso
            inicio_segmento = pico + 1  # Asumiendo que el impulso termina justo después del pico
        
        # Añadir el último segmento después del último pico, si existe
        if inicio_segmento < len(senal):
            segmentos.append(senal[inicio_segmento:])
    
    return segmentos

def dividir_señales_en_ciclos(abf, canal, fs):
    """
    Divide una señal en ciclos basándose en su frecuencia dominante.

    :param abf: Objeto ABF cargado.
    :param canal: Índice del canal a procesar.
    :param fs: Frecuencia de muestreo de la señal.
    :return: Lista de segmentos de la señal, cada uno correspondiendo a un ciclo.
    """
    abf.setSweep(sweepNumber=0, channel=canal)
    senal = abf.sweepY
    
    # Encontrar la frecuencia dominante
    frecuencia_dominante = encontrar_frecuencia_dominante(senal, fs)
    periodo = 1 / frecuencia_dominante  # Duración de un ciclo en segundos
    
    # Calcular el número de muestras por ciclo
    muestras_por_ciclo = int(np.round(periodo * fs))
    
    # Dividir la señal en segmentos basados en el número estimado de muestras por ciclo
    segmentos = [senal[i:i+muestras_por_ciclo] for i in range(0, len(senal), muestras_por_ciclo)]
    
    return segmentos

def encontrar_frecuencia_dominante(senal, fs):
    """
    Encuentra la frecuencia dominante de una señal utilizando FFT.

    :param senal: Array de NumPy con los datos de la señal.
    :param fs: Frecuencia de muestreo de la señal.
    :return: La frecuencia dominante en la señal.
    """
    # Aplicar FFT a la señal
    fft_res = np.fft.fft(senal)
    freqs = np.fft.fftfreq(len(senal), d=1/fs)
    
    # Encontrar el índice de la frecuencia con la mayor magnitud en la FFT (excluyendo la componente de DC)
    idx = np.argmax(np.abs(fft_res[1:])) + 1
    frecuencia_dominante = abs(freqs[idx])
    
    return frecuencia_dominante

def canal_tiene_picos(abf, canal, voltaje_umbral=5):
    abf.setSweep(sweepNumber=0, channel=canal)
    senal = abf.sweepY
    picos, _ = find_peaks(senal, height=voltaje_umbral)
    return len(picos) > 0

def dividir_señales_por_tiempo(abf, ventana_tiempo_ms=290):
    """
    Divide las señales de un objeto ABF en segmentos basados en una ventana de tiempo.
    
    :param abf: Objeto ABF cargado.
    :param ventana_tiempo_ms: Ventana de tiempo en milisegundos para dividir las señales.
    :return: Lista de diccionarios, cada uno representando segmentos de señales con sus metadatos.
    """
    fs = abf.dataRate  # Frecuencia de muestreo en Hz
    muestras_por_segmento = int((ventana_tiempo_ms / 1000) * fs)  # Número de muestras por segmento
    
    segmentos = []
    for canal in range(abf.channelCount):
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweepNumber=sweep, channel=canal)
            for i in range(0, abf.sweepPointCount, muestras_por_segmento):
                # Asegurarse de no exceder el límite de la señal
                fin_segmento = min(i + muestras_por_segmento, abf.sweepPointCount)
                # Extraer segmento
                datos_segmento = abf.sweepY[i:fin_segmento]
                tiempo_segmento = abf.sweepX[i:fin_segmento]
                # Guardar segmento y metadatos
                segmentos.append({
                    'canal': canal,
                    'sweep': sweep,
                    'datos': datos_segmento,
                    'tiempo': tiempo_segmento,
                    'metadatos': {
                        'unidad_medida': abf.adcUnits[canal],
                        'frecuencia_muestreo': fs,
                        'inicio_ms': i / fs * 1000,
                        'fin_ms': fin_segmento / fs * 1000
                    }
                })
    
    return segmentos

def dividir_señales_por_tiempo_y_excluir_picos(abf, ventana_tiempo_ms=290, voltaje_umbral=5, distancia_minima_entre_impulsos=1000):
    """
    Divide las señales de un objeto ABF en segmentos basados en una ventana de tiempo y excluye segmentos que contienen picos de impulsos.
    
    :param abf: Objeto ABF cargado.
    :param ventana_tiempo_ms: Ventana de tiempo en milisegundos para dividir las señales.
    :param voltaje_umbral: Umbral de voltaje para detectar impulsos.
    :param distancia_minima_entre_impulsos: Distancia mínima en índices entre picos consecutivos.
    :return: Lista de diccionarios, cada uno representando segmentos de señales sin picos, con sus metadatos.
    """
    fs = abf.dataRate  # Frecuencia de muestreo en Hz
    muestras_por_segmento = int((ventana_tiempo_ms / 1000) * fs)  # Número de muestras por segmento
    
    segmentos = []
    for canal in range(abf.channelCount):
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweepNumber=sweep, channel=canal)
            
            # Detectar los picos (impulsos) en toda la señal de este sweep
            picos, _ = find_peaks(abf.sweepY, height=voltaje_umbral, distance=distancia_minima_entre_impulsos)
            
            for i in range(0, abf.sweepPointCount, muestras_por_segmento):
                fin_segmento = min(i + muestras_por_segmento, abf.sweepPointCount)
                
                # Verificar si el segmento actual contiene alguno de los picos detectados
                if not np.any((picos >= i) & (picos < fin_segmento)):
                    # Si no contiene picos, extraer segmento
                    datos_segmento = abf.sweepY[i:fin_segmento]
                    tiempo_segmento = abf.sweepX[i:fin_segmento]
                    # Guardar segmento y metadatos
                    segmentos.append({
                        'canal': canal,
                        'sweep': sweep,
                        'datos': datos_segmento,
                        'tiempo': tiempo_segmento,
                        'metadatos': {
                            'unidad_medida': abf.adcUnits[canal],
                            'frecuencia_muestreo': fs,
                            'inicio_ms': i / fs * 1000,
                            'fin_ms': fin_segmento / fs * 1000
                        }
                    })
    
    return segmentos

def guardar_segmentos_y_metadatos(segmentos, metadatos, ruta_directorio):
    """
    Guarda los segmentos de señales en archivos CSV y metadatos en un archivo JSON en la ruta especificada.
    No guarda un archivo si ya existe con el mismo nombre.
    
    :param segmentos: Lista de diccionarios, cada uno representando segmentos de señales con sus metadatos.
    :param metadatos: Diccionario con metadatos adicionales de la señal.
    :param ruta_directorio: Ruta del directorio donde se guardarán los archivos.
    """
    # Guardar metadatos en un archivo JSON solo si no existe previamente
    ruta_metadatos = os.path.join(ruta_directorio, "metadatos.json")
    if not os.path.exists(ruta_metadatos):
        with open(ruta_metadatos, 'w') as file:
            json.dump(metadatos, file, indent=4)
        print(f"Metadatos guardados: {ruta_metadatos}")
    else:
        print(f"Archivo de metadatos ya existe: {ruta_metadatos} (No guardado)")

    # Guardar cada segmento en un archivo CSV solo si no existe previamente
    for i, segmento in enumerate(segmentos):
        df = pd.DataFrame({
            'Tiempo (s)': segmento['tiempo'],
            'Datos': segmento['datos']
        })
        
        nombre_archivo = f"segmento_{i}_canal_{segmento['canal']}_{segmento['metadatos']['inicio_ms']:.0f}-{segmento['metadatos']['fin_ms']:.0f}ms.csv"
        ruta_completa = os.path.join(ruta_directorio, nombre_archivo)
        
        if not os.path.exists(ruta_completa):
            df.to_csv(ruta_completa, index=False)
            print(f"Guardado segmento: {ruta_completa}")
        else:
            print(f"Archivo de segmento ya existe: {ruta_completa} (No guardado)")

def procesar_archivos_abf(carpeta_abf, carpeta_destino, intervalo_ms=290):
    """
    Procesa todos los archivos .abf en una carpeta, extrayendo metadatos, detectando picos de voltaje,
    dividiendo las señales con picos en segmentos y guardando los resultados.
    """
    os.makedirs(carpeta_destino, exist_ok=True)

    for archivo in os.listdir(carpeta_abf):
        if archivo.endswith('.abf'):
            ruta_archivo = os.path.join(carpeta_abf, archivo)
            print(ruta_archivo)
            abf = pyabf.ABF(ruta_archivo)
            metadatos = extraer_metadatos(abf)  # Extraer metadatos aquí para verificar después si guardarlo
            print(metadatos)
            canales_con_segmentos = False  # Para verificar si al menos un canal tiene segmentos válidos

            for canal in range(abf.channelCount):
                print(canal_tiene_picos(abf, canal, voltaje_umbral=5))
                if canal_tiene_picos(abf, canal, voltaje_umbral=5):
                    segmentos = dividir_señales_por_tiempo_y_excluir_picos(abf, ventana_tiempo_ms=intervalo_ms, voltaje_umbral=5, distancia_minima_entre_impulsos=1000)
                    if segmentos:
                        canales_con_segmentos = True
                        carpeta_canal = os.path.join(carpeta_destino, os.path.splitext(archivo)[0], f'canal_{canal}')
                        os.makedirs(carpeta_canal, exist_ok=True)
                        guardar_segmentos_y_metadatos(segmentos, metadatos, carpeta_canal)
            
            # Guardar metadatos solo si al menos un canal tenía segmentos válidos
            if canales_con_segmentos:
                nombre_sin_extension = os.path.splitext(archivo)[0]
                ruta_metadatos = os.path.join(carpeta_destino, f'{nombre_sin_extension}_metadatos.json')
                with open(ruta_metadatos, 'w') as archivo_json:
                    json.dump(metadatos, archivo_json, indent=4)
