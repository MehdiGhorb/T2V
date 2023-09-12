'''
Download the videos by providing the csv path to the URLs and a path to save the videos
Run this in Shell or use Jupyter notebook:
python3 your_script.py /path/to/your.csv /path/to/save/videos 10
'''

import argparse
import os
import yaml
import sys
sys.path.append('../helper')
from dataLoader import *

base_directory = '../original_videos'
base_data_dir = '../data/'

with open('../data/track.yaml', 'w') as yaml_file:
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)
total_number_of_videos = data['total_videos_checkpoint']

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    parser.add_argument('num_videos', type=int, help='Number of videos per iteration')

    args = parser.parse_args()

    # Use os.listdir() to get a list of all files and directories in the specified path
    contents = os.listdir(base_directory)

    # Filter the list to only include directories
    directories = [item for item in contents if os.path.isdir(os.path.join(base_directory, item))]

    # Print the list of directories
    for directory in directories:
        print(directory)


    createDirectory(f"../original_videos/video_{args.num_videos}")

    rows = read_data(base_data_dir + args.csv_file_name, start_index=total_number_of_videos, end_index=args.num_videos + total_number_of_videos)
    video_urls = []

    for index in rows:
        video_urls.append(index[1])

    indexes_to_delete = download_videos(video_urls, base_directory+f"/{len(indexes)}_video_{args.num_videos}")

#Update the yaml file
for i in indexes_to_delete:
    data['failed_indexes'].append(i)

data['failed_indexes'] += total_videos_checkpoint

if __name__ == "__main__":
    main()
    