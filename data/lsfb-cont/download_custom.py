import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from lsfb_dataset import Downloader

# Path to the CSV file
csv_file_path = './dataset/instances.csv'

# Directory where the files will be saved
destination_directory = './dataset/videos/'

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)


# Function to download the file
def download_file(file_id):
    url = f"https://lsfb.info.unamur.be/static/datasets/lsfb_v2/cont/videos/{file_id}.mp4"
    command = f"wget --continue -P {destination_directory} {url}"
    subprocess.run(command, shell=True)


# Process the CSV and download files
def process_files():
    downloader = Downloader(dataset='cont', destination="./dataset", landmarks=[], include_videos=False,
                            max_parallel_connections=10, skip_existing_files=True)
    downloader.download()

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        # Create a list of file IDs
        file_ids = [row[0] for row in reader if row]  # Ensures no empty rows are processed

    # Using ThreadPoolExecutor to download files concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(download_file, file_ids)
        for result in results:
            pass  # This loop ensures we wait for all downloads to complete


# Run the processing function
if __name__ == "__main__":
    process_files()
