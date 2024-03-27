import pyabf
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from scipy.signal import find_peaks
import os
import pandas as pd
import json
from IPython.display import display
from scipy import signal

def normalize_and_segment_channel(abf_file_name, channel_number, normalization_type='z_score', source_directory='data/segments', destination_directory='data/segments_normalized', segment_duration_ms=290, sampling_rate=1000):
    """
    Normalizes and segments data for a specific channel from .csv files based on an ABF file name, then saves the segments into a new directory.
    
    Parameters:
    - abf_file_name: str, name of the original ABF file without the extension.
    - channel_number: int, the specific channel to process.
    - normalization_type: str, type of normalization ('z_score', 'min_max', or 'none').
    - source_directory: str, path to the directory containing the original segments.
    - destination_directory: str, path to the directory where normalized segments will be saved.
    - segment_duration_ms: int, duration of each segment in milliseconds.
    - sampling_rate: int, sampling rate of the data in Hz.
    """
    # Ensure the destination directory exists
    destination_path = os.path.join(destination_directory, abf_file_name)
    os.makedirs(destination_path, exist_ok=True)

    pattern = f"segmento_*_canal_{channel_number}_"
    print(f"Searching for files matching the pattern: {pattern} within {abf_file_name}")

    files_to_process = []
    # Now we adjust the source_path to include the channel directory
    source_path = os.path.join(source_directory, abf_file_name, f"canal_{channel_number}")
    print(f"Looking in: {source_path}")
    
    # The pattern adjusted to not specifically include 'canal_X' since we're already in the specific channel directory
    pattern = "segmento_"
    print(f"Searching for files matching the pattern: {pattern} within {abf_file_name}, channel {channel_number}")

    files_to_process = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if pattern in file:
                print("File to process:", os.path.join(root, file))
                files_to_process.append(os.path.join(root, file))

    print("____________________________________________________________________________")
    # Sort files to process in ascending order based on segment number or time interval
    files_to_process.sort(key=lambda x: int(x.split('_')[1]))

    all_data = []
    for file_path in files_to_process:
        df = pd.read_csv(file_path)
        all_data.append(df['Data'].values)
    
    # Concatenate all data segments
    full_signal = np.concatenate(all_data)
    
    # Normalize the full signal
    if normalization_type == 'z_score':
        normalized_signal = (full_signal - np.mean(full_signal)) / np.std(full_signal)
    elif normalization_type == 'min_max':
        normalized_signal = (full_signal - np.min(full_signal)) / (np.max(full_signal) - np.min(full_signal))
    elif normalization_type == 'none':
        normalized_signal = full_signal
    else:
        raise ValueError("Invalid normalization type. Use 'z_score', 'min_max', or 'none'.")

    # Segment the normalized signal
    segment_length = int((segment_duration_ms / 1000) * sampling_rate)
    num_segments = len(normalized_signal) // segment_length

    for i in range(num_segments):
        segment = normalized_signal[i*segment_length:(i+1)*segment_length]
        segment_df = pd.DataFrame(segment, columns=['Normalized Data'])
        
        # Construct output file name based on segment time
        start_time_ms = i * segment_duration_ms
        end_time_ms = (i + 1) * segment_duration_ms
        segment_file_name = f"{abf_file_name}_normalized_segment_{i}_channel_{channel_number}_{start_time_ms}_{end_time_ms}.csv"
        segment_file_path = os.path.join(destination_path, segment_file_name)
        
        # Save the segment to a CSV file
        segment_df.to_csv(segment_file_path, index=False)
        print(f"Segment saved: {segment_file_path}")

    print("Normalization and segmentation completed.")


def determine_recording_mode(abf):
    """
    Determines the recording mode of an ABF file.

    :param abf: Loaded ABF object.
    :return: String describing the recording mode.
    """
    if abf.sweepCount > 1:
        episode_lengths = [abf.sweepLengthInPoints(i) for i in range(abf.sweepCount)]
        if len(set(episode_lengths)) == 1:
            return "Fixed Episode: Data collected in episodes of fixed length."
        else:
            return "Variable Episode: Data collected in episodes, but with variable lengths."
    else:
        return "Gap-Free: A continuous flow of data."

def extract_metadata(abf):
    """
    Extracts metadata from a loaded ABF object.

    :param abf: Loaded ABF object.
    :return: Dictionary with ABF object metadata.
    """
    metadata = {
        "file_name": abf.abfID,
        "sampling_rate": abf.dataRate,
        "channel_count": abf.channelCount,
        "episode_count": abf.sweepCount,
        "total_time": abf.sweepCount / abf.dataRate,
        "recording_date": abf.abfDateTimeString,
        "comments": abf.abfFileComment,
        "time_unit": "seconds",
        "channel_units": {},
        "channel_labels": {},
        "data_dimensions": abf.data.shape
    }

    for i in range(abf.channelCount):
        abf.setSweep(0, channel=i)
        metadata["channel_units"][i] = abf.adcUnits
        metadata["channel_labels"][i] = abf.adcNames

    return metadata

def select_data(abf, start=0, end=None, episodes='all', channels='all'):
    """
    Selects specific segments of data in an ABF file.

    :param abf: Loaded ABF object.
    :param start: Start of the time segment for data selection (in seconds).
    :param end: End of the time segment for data selection (in seconds).
    :param episodes: Specific episodes to read. Can be a number, a list, or 'all'.
    :param channels: Specific channels to read. Can be a number, a list, or 'all'.
    :return: Tuple with the time and selected data.
    """
    selected_data = {}
    sampling_rate = abf.dataRate

    start_point = int(start * sampling_rate)
    end_point = int(end * sampling_rate) if end is not None else None

    episodes_to_read = range(abf.sweepCount) if episodes == 'all' else episodes
    channels_to_read = range(abf.channelCount) if channels == 'all' else channels

    for i in episodes_to_read:
        abf.setSweep(sweepNumber=i)
        time = abf.sweepX[start_point:end_point]
        for channel in channels_to_read:
            abf.setSweep(sweepNumber=i, channel=channel)
            selected_data[(i, channel)] = abf.sweepY[start_point:end_point]

    return time, selected_data

def plot_channels_interactively(time, data, metadata, title="ABF Data by Channel"):
    """
    Plots data from each channel of an ABF file interactively,
    using the metadata for the units.
    """
    def update_plot(episode=None, channel=None):
        plt.clf()
        if episode is not None and channel is not None:
            key = (episode, channel)
            values = data[key]
            unit = metadata["channel_units"].get(channel, "Unknown Unit")
            plt.plot(time, values)
            plt.title(f"{title} - Episode {episode}, Channel {channel}")
            plt.ylabel(unit)
        else:
            for key, values in data.items():
                episode, channel = key
                unit = metadata["channel_units"].get(channel, "Unknown Unit")
                plt.plot(time, values, label=f"Episode {episode}, Channel {channel} ({unit})")
            plt.title(f"{title} - All Channels")
            plt.legend()

        plt.xlabel("Time (s)")
        plt.show()

    episodes = list({key[0] for key in data.keys()})
    channels = list({key[1] for key in data.keys()})
    episode_widget = widgets.Dropdown(options=[None] + episodes, description='Episode:')
    channel_widget = widgets.Dropdown(options=[None] + channels, description='Channel:')

    def update_channel(*args):
        episode = episode_widget.value
        channel = channel_widget.value
        update_plot(episode, channel)

    episode_widget.observe(update_channel, 'value')
    channel_widget.observe(update_channel, 'value')

    display(episode_widget, channel_widget)
    update_plot()

def extract_segments_between_spikes(abf, channel, voltage_threshold=5, minimum_spike_distance=1000):
    """
    Extracts and returns segments of a signal that are between spikes defined by a voltage threshold,
    for a specific channel of a multi-channel ABF signal.

    :param abf: Loaded ABF object, e.g., using pyABF.
    :param channel: Zero-based index of the channel to process.
    :param voltage_threshold: Voltage threshold to detect spikes.
    :param minimum_spike_distance: Minimum distance in indices between consecutive spikes.
    :return: List of NumPy arrays, each representing a signal segment between spikes for the given channel.
    """
    segments = []

    for i in range(abf.sweepCount):
        abf.setSweep(i, channel=channel)
        signal = abf.sweepY
        spikes, _ = find_peaks(signal, height=voltage_threshold, distance=minimum_spike_distance)

        start_segment = 0
        for spike in spikes:
            segments.append(signal[start_segment:spike])
            start_segment = spike + 1

        if start_segment < len(signal):
            segments.append(signal[start_segment:])

    return segments

def divide_signals_into_cycles(abf, channel, fs):
    """
    Divides a signal into cycles based on its dominant frequency.

    :param abf: Loaded ABF object.
    :param channel: Index of the channel to process.
    :param fs: Sampling rate of the signal.
    :return: List of signal segments, each corresponding to a cycle.
    """
    abf.setSweep(sweepNumber=0, channel=channel)
    signal = abf.sweepY

    dominant_frequency = find_dominant_frequency(signal, fs)
    period = 1 / dominant_frequency

    samples_per_cycle = int(np.round(period * fs))
    segments = [signal[i:i+samples_per_cycle] for i in range(0, len(signal), samples_per_cycle)]

    return segments

def find_dominant_frequency(signal, fs):
    """
    Finds the dominant frequency of a signal using FFT.

    :param signal: NumPy array with signal data.
    :param fs: Sampling rate of the signal.
    :return: The dominant frequency in the signal.
    """
    fft_res = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    idx = np.argmax(np.abs(fft_res[1:])) + 1
    dominant_frequency = abs(freqs[idx])

    return dominant_frequency

def channel_has_spikes(abf, channel, voltage_threshold=5):
    """
    Checks if a channel has spikes exceeding a voltage threshold.

    :param abf: Loaded ABF object.
    :param channel: Index of the channel to check.
    :param voltage_threshold: Voltage threshold to detect spikes.
    :return: Boolean indicating if the channel has spikes.
    """
    abf.setSweep(sweepNumber=0, channel=channel)
    signal = abf.sweepY
    spikes, _ = find_peaks(signal, height=voltage_threshold)
    return len(spikes) > 0

def divide_signals_by_time(abf, time_window_ms=290):
    """
    Divides the signals of an ABF object into segments based on a time window.

    :param abf: Loaded ABF object.
    :param time_window_ms: Time window in milliseconds to divide the signals.
    :return: List of dictionaries, each representing signal segments with their metadata.
    """
    fs = abf.dataRate
    samples_per_segment = int((time_window_ms / 1000) * fs)
    
    segments = []
    for channel in range(abf.channelCount):
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweepNumber=sweep, channel=channel)
            for i in range(0, abf.sweepPointCount, samples_per_segment):
                end_segment = min(i + samples_per_segment, abf.sweepPointCount)
                data_segment = abf.sweepY[i:end_segment]
                time_segment = abf.sweepX[i:end_segment]
                segments.append({
                    'channel': channel,
                    'sweep': sweep,
                    'data': data_segment,
                    'time': time_segment,
                    'metadata': {
                        'unit_measure': abf.adcUnits[channel],
                        'sampling_rate': fs,
                        'start_ms': i / fs * 1000,
                        'end_ms': end_segment / fs * 1000
                    }
                })

    return segments

def divide_signals_by_time_and_exclude_spikes(abf, time_window_ms=290, voltage_threshold=5, minimum_spike_distance=1000, channel_number=0):
    """
    Divide las señales de un objeto ABF en segmentos basados en una ventana de tiempo y excluye segmentos que contienen impulsos de pico.

    :param abf: Objeto ABF cargado.
    :param time_window_ms: Ventana de tiempo en milisegundos para dividir las señales.
    :param voltage_threshold: Umbral de voltaje para detectar impulsos.
    :param minimum_spike_distance: Distancia mínima en índices entre picos consecutivos.
    :param channel_number: Número del canal a procesar.
    :return: Lista de diccionarios, cada uno representando segmentos de señal sin picos, con sus metadatos.
    """
    fs = abf.dataRate
    samples_per_segment = int((time_window_ms / 1000) * fs)
    
    segments = []
    abf.setSweep(sweepNumber=0, channel=channel_number)
    spikes, _ = find_peaks(abf.sweepY, height=voltage_threshold, distance=minimum_spike_distance)
    
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweepNumber=sweep, channel=channel_number)
        
        for i in range(0, abf.sweepPointCount, samples_per_segment):
            end_segment = min(i + samples_per_segment, abf.sweepPointCount)
            if not np.any((spikes >= i) & (spikes < end_segment)):
                data_segment = abf.sweepY[i:end_segment]
                time_segment = abf.sweepX[i:end_segment]
                segments.append({
                    'channel': channel_number,
                    'sweep': sweep,
                    'data': data_segment,
                    'time': time_segment,
                    'metadata': {
                        'unit_measure': abf.adcUnits[channel_number],
                        'sampling_rate': fs,
                        'start_ms': i / fs * 1000,
                        'end_ms': end_segment / fs * 1000
                    }
                })

    return segments

def save_segments_and_metadata(segments, metadata, directory_path):
    """
    Saves signal segments in CSV files and metadata in a JSON file at the specified path.
    Does not save a file if one already exists with the same name.

    :param segments: List of dictionaries, each representing signal segments with their metadata.
    :param metadata: Dictionary with additional signal metadata.
    :param directory_path: Path of the directory where the files will be saved.
    """
    metadata_path = os.path.join(directory_path, "metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        print(f"Metadata saved: {metadata_path}")
    else:
        print(f"Metadata file already exists: {metadata_path} (Not saved)")

    for i, segment in enumerate(segments):
        df = pd.DataFrame({
            'Time (s)': segment['time'],
            'Data': segment['data']
        })
        
        file_name = f"segment_{i}_channel_{segment['channel']}_{segment['metadata']['start_ms']:.0f}-{segment['metadata']['end_ms']:.0f}ms.csv"
        full_path = os.path.join(directory_path, file_name)
        
        if not os.path.exists(full_path):
            df.to_csv(full_path, index=False)
            print(f"Segment saved: {full_path}")
        else:
            print(f"Segment file already exists: {full_path} (Not saved)")


def process_abf_files(source_folder, destination_folder, time_window_ms=290):
    """
    Processes all .abf files in a folder, extracting metadata, detecting voltage spikes,
    dividing signals with spikes into segments, and saving the results.

    :param source_folder: Folder containing .abf files to process.
    :param destination_folder: Folder where processed files will be saved.
    :param time_window_ms: Time window in milliseconds for dividing signals.
    """
    os.makedirs(destination_folder, exist_ok=True)  # Asegurar que el directorio de destino existe

    for file in os.listdir(source_folder):
        if file.endswith('.abf'):  # Procesar solo archivos .abf
            file_path = os.path.join(source_folder, file)
            abf = pyabf.ABF(file_path)  # Cargar archivo ABF
            metadata = extract_metadata(abf)  # Extraer metadatos

            for channel in range(abf.channelCount):  # Iterar sobre todos los canales
                if channel_has_spikes(abf, channel, voltage_threshold=5):  # Verificar si el canal tiene picos
                    # Dividir señales por tiempo y excluir segmentos con picos
                    segments = divide_signals_by_time_and_exclude_spikes(abf, time_window_ms=time_window_ms, voltage_threshold=5, minimum_spike_distance=1000, channel_number = channel)
                    if segments:
                        # Crear carpeta específica para este canal
                        channel_folder = os.path.join(destination_folder, os.path.splitext(file)[0], f'channel_{channel}')
                        os.makedirs(channel_folder, exist_ok=True)  # Asegurar que la carpeta del canal existe
                        save_segments_and_metadata(segments, metadata, channel_folder)  # Guardar segmentos y metadatos
