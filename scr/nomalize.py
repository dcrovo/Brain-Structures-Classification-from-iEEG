from functions import normalize_and_segment_channel

# Function parameters
abf_file_name = '15o14000'  # The original ABF file name without the extension
channel_number = 1  # The specific channel you want to process
source_directory = 'data/segments'  # Path to the directory containing the original segments
destination_directory = 'data/segments_normalized'  # Path where the normalized segments will be saved
segment_duration_ms = 290  # Duration of each segment in milliseconds
sampling_rate = 1000  # Sampling rate of the data in Hz

# Calling the function
normalize_and_segment_channel(abf_file_name, channel_number, 'z_score', source_directory, destination_directory, segment_duration_ms, sampling_rate)