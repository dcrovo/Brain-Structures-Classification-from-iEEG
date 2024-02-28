import pyabf
import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que las funciones necesarias están definidas en un archivo llamado abf_functions.py
from functions import extract_metadata, select_data

def plot_abf_channels(abf_file_name):
    """
    Load an ABF file by its name, extract data and metadata, and plot the signals of each channel.
    Assumes that ABF files are stored in 'data/source/' directory.

    :param abf_file_name: Name of the ABF file (with extension).
    """
    abf_file_path = f'data/source/{abf_file_name}'
    abf = pyabf.ABF(abf_file_path)
    metadata = extract_metadata(abf)
    
    # Use select_data to get data from all channels and episodes
    time, data = select_data(abf, start=0, end=None, episodes='all', channels='all')

    # Plot data for each channel
    for channel in range(metadata["channel_count"]):
        plt.figure(figsize=(10, 4))
        plt.title(f"Channel {channel}: {metadata['channel_labels'][channel]}")
        plt.xlabel("Time (s)")
        plt.ylabel(metadata["channel_units"][channel])
        
        # Plot all episodes for this channel
        for episode in range(metadata["episode_count"]):
            if (episode, channel) in data:
                plt.plot(time, data[(episode, channel)], label=f"Episode {episode}")
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
abf_file_name = "15o14000.abf"  # Replace with your ABF file name
plot_abf_channels(abf_file_name)
