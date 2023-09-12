'''
Download the videos by providing the csv path to the URLs and a path to save the videos
Run this in Shell or use Jupyter notebook:
python3 your_script.py /path/to/your.csv /path/to/save/videos 10
'''

import argparse
import os
import sys
sys.path.append('../helper')
from dataLoader import *

base_directory = '../original_videos'
base_data_dir = '../data/'

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    parser.add_argument('iteration', help='Number of iterations for downloading videos')
    parser.add_argument('num_videos', type=int, help='Number of videos per iteration')

    args = parser.parse_args()

    # Use os.listdir() to get a list of all files and directories in the specified path
    contents = os.listdir(base_directory)

    # Filter the list to only include directories
    directories = [item for item in contents if os.path.isdir(os.path.join(base_directory, item))]

    # Print the list of directories
    for directory in directories:
        print(directory)

    for _ in range(int(args.iteration)):

        indexes = []
        total_number_of_videos = 0
        # Iterate through the strings and extract the numeric part
        for name in directories:
            parts = name.split("_")

            if parts[0].isdigit():
                indexes.append(int(parts[0]))
            if parts[2].isdigit():
                total_number_of_videos += int(parts[2])


        createDirectory(f"../original_videos/{len(indexes)}_video_{args.num_videos}")


        rows = read_data(base_data_dir + args.csv_file_name, start_index=total_number_of_videos, end_index=args.num_videos + total_number_of_videos)
        video_urls = []

        for index in rows:
            video_urls.append(index[1])

        indexes_to_delete = download_videos(video_urls, base_directory+f"/{len(indexes)}_video_{args.num_videos}")

if __name__ == "__main__":
    main()
    