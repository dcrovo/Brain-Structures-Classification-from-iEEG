import pyabf
import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que las funciones necesarias están definidas en otro lugar
from functions import extract_metadata, select_data

def plot_abf_channels(abf_file_name):
    """
    Carga un archivo ABF por su nombre, extrae datos y metadatos, y grafica las señales de cada canal.
    Asume que los archivos ABF se almacenan en el directorio 'data/source/'.

    :param abf_file_name: Nombre del archivo ABF (con extensión).
    """
    abf_file_path = f'data/source/{abf_file_name}'
    abf = pyabf.ABF(abf_file_path)
    metadata = extract_metadata(abf)
    print(metadata)
    
    # Usar select_data para obtener datos de todos los canales y episodios
    time, data = select_data(abf, start=0, end=None, episodes='all', channels='all')

    # Crear una figura y un conjunto de subgráficas
    fig, axs = plt.subplots(metadata["channel_count"], 1, figsize=(10, 4 * metadata["channel_count"]), sharex=True)

    if metadata["channel_count"] == 1:
        axs = [axs]  # Asegurar que axs sea iterable incluso si hay solo un canal

    for channel, ax in enumerate(axs):
        ax.set_title(f"Channel {channel}: {metadata['channel_labels'][channel]}")
        ax.set_ylabel(metadata["channel_units"][channel])

        # Graficar todos los episodios para este canal
        for episode in range(metadata["episode_count"]):
            if (episode, channel) in data:
                ax.plot(time, data[(episode, channel)], label=f"Episode {episode}")

        ax.legend()
    
    axs[-1].set_xlabel("Time (s)")  # Solo añadir etiqueta x al último subplot
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
abf_file_name = "15o14000.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "15o14006.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "15o14024.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "15o14031.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17613000.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17613006.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17613025.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17613029.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17620000.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17620007.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17620028.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)

abf_file_name = "17620042.abf"  # Reemplazar con el nombre de tu archivo ABF
plot_abf_channels(abf_file_name)
