import sys
sys.path.append('/helper')
from videoAnnotation import videoToGIF
import csv
import requests
from tqdm import tqdm
import os

def read_data(csv_file_path, videos=20000):
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file)

        # Skip the header row if it exists
        header = next(csv_reader, None)

        # Display the first 10 rows
        rows = []
        for _ in range(videos):
            row = next(csv_reader, None)
            if row is not None:
                rows.append(row)
        return rows

def extract_duration(iso_format):
  duration = parse_duration(iso_format)
  duration_seconds = duration.total_seconds()
  return duration_seconds

def download_videos(video_urls: list[str], dir_path):
  failed_downloads = []
  j=0
  for video_url in tqdm(video_urls, desc='Downloading videos'):
      response = requests.get(video_url)  # Send a GET request to download the video
      http_status = response.status_code

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
  video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
  video_files = [os.path.join(directory, file) for file in all_files if any(file.lower().endswith(ext) for ext in video_extensions)]

  sorted_video_files = sorted(video_files, key=extract_numeric_part)

  # Store the video file names with directory paths in a list
  return sorted_video_files

def createDirectory(directory):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
