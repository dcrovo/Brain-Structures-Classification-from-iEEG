from functions import normalize_and_segment_channel
import os
import pandas as pd
import numpy as np

def normalize_matrix(data, mode='z_score'):
    if mode == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
    elif mode == 'min_max':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif mode == 'max_abs':
        max_abs = np.max(np.abs(data))
        normalized_data = data / max_abs
    else:
        raise ValueError("Unsupported normalization mode. Choose 'z_score', 'min_max', or 'max_abs'.")
    return normalized_data

def process_files(source_directory, destination_directory, normalization_mode='z_score'):
    # Asegurarse de que el directorio de destino existe
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterar sobre cada archivo en el directorio fuente
    for filename in os.listdir(source_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_directory, filename)
            df = pd.read_csv(file_path)
            
            # Asumiendo que la matriz de datos está correctamente formatada en el DataFrame
            data_matrix = df.values
            normalized_matrix = normalize_matrix(data_matrix, mode=normalization_mode)
            
            # Convertir la matriz normalizada de nuevo a DataFrame
            normalized_df = pd.DataFrame(normalized_matrix, columns=df.columns)
            
            # Guardar el DataFrame normalizado en el directorio destino
            destination_file_path = os.path.join(destination_directory, filename)
            normalized_df.to_csv(destination_file_path, index=False)
            print(f"File processed and saved: {destination_file_path}")

# Rutas de los directorios
source_dir = r'C:\Users\franc\Downloads\saved_matrix\saved_matrix'
destination_dir = r'C:\Users\franc\Downloads\saved_matrix\saved_matrix_normalize'

# Llamar a la función de procesamiento
process_files(source_dir, destination_dir)
