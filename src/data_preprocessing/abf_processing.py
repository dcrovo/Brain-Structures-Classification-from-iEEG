from functions import process_abf_files

# Define the source and destination folder paths
source_folder_abf = './data/source'
destination_folder_segments = './data/segments'

# Call the function to process the .abf files
process_abf_files(source_folder_abf, destination_folder_segments)