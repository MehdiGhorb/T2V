import sys
sys.path.append('/helper')
from videoAnnotation import videoToGIF
import csv
import requests
from tqdm import tqdm
import os
import glob
import shutil

def read_data(csv_file_path, start_index=0, end_index=10000):
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file)
        # Read the header row if it exists
        skip_header = next(csv_reader, None)
        # Skip rows until the start index
        for _ in range(start_index):
            next(csv_reader, None)
        # Read and return rows within the specified interval, while handling empty rows
        rows = []
        for row in csv_reader:
            if row and start_index <= end_index:
                rows.append(row)
                start_index += 1
        return rows

def extract_duration(iso_format):
    duration = parse_duration(iso_format)
    duration_seconds = duration.total_seconds()
    return duration_seconds

def download_videos(video_urls: list[str], dir_path):
    failed_downloads = []
    j=2
    for video_url in tqdm(video_urls, desc='Downloading videos'):
        try:
            response = requests.get(video_url)  # Send a GET request to download the video
            http_status = response.status_code
        #except requests.exceptions.ConnectionError or requests.exceptions.InvalidURL:
        except:
            http_status = 404

        if http_status == 200:
            video_path = dir_path + f"/video_{video_urls.index(video_url)}.mp4"

            with open(video_path, "wb") as f:
                f.write(response.content)

        else:
            print(f"Download failed for {video_url} (HTTP {http_status})")
            failed_downloads.append(j)
        j+=1
    return failed_downloads

# Sort the video files based on the numeric part of the file names
def extract_numeric_part(filename):
    return int(filename.split('_')[-1].split('.')[0])

def getVideoNames(directory):

  # Get a list of all files in the directory
  all_files = os.listdir(directory)

  # Filter out video files (you can customize the extensions as needed)
  video_extensions = ['.mp4', ".gif"]
  video_files = [os.path.join(directory, file) for file in all_files if any(file.lower().endswith(ext) for ext in video_extensions)]

  sorted_video_files = sorted(video_files, key=extract_numeric_part)

  # Store the video file names with directory paths in a list
  return sorted_video_files

def createDirectory(directory):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

def removeDirContent(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Iterate through all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            # Delete the file
            os.unlink(item_path)
        # Check if the item is a subdirectory
        elif os.path.isdir(item_path):
            # Delete the subdirectory and its contents recursively
            shutil.rmtree(item_path)

def countVideosInDirectory(directory_path):
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    # Use the glob module to search for video files (you can add more extensions as needed)
    video_extensions = ["*.mp4", ".gif"]
    video_files = []
    
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory_path, extension)))

    # Return the count of video files found
    return len(video_files)

def createCustomedVideoCsvFile(video_urls, video_names, csv_filename):
    if len(video_urls) != len(video_names):
        raise ValueError("Length of video_urls and video_names must be the same.")

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Video URL', 'Video Name'])
        
        for url, name in zip(video_urls, video_names):
            csv_writer.writerow([url, name])
            