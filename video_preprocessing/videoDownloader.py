'''
Download the videos by providing the csv path to the URLs and a path to save the videos
Run this in Shell or use Jupyter notebook:
python3 your_script.py /path/to/your.csv /path/to/save/videos 10
'''

import argparse
from yamlEditor import *
import sys
sys.path.append('../helper')
from dataLoader import *

base_directory = '../original_videos'
base_data_dir = '../data/'
base_logs_dir = '../logs/'

#keys = extractKeysFromYaml(data)
#largest = findLargestIndexedString(data)
#print(largest)
# Write the updated data back to the YAML file
#with open(track_file, 'w') as yaml_file:
#    yaml.dump(data, yaml_file)

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    parser.add_argument('num_videos', type=int, help='Number of videos per iteration')
    args = parser.parse_args()

    # Read data from main YAML
    temp = args.csv_file_name.replace('.csv', '')
    main_yaml = loadMainYamlFile(base_logs_dir + f"track_{temp}.yaml")
    total_number_of_videos = main_yaml['Total_video_checkpoint']
    # Update main YAML values
    main_yaml['Total_video_checkpoint'] = main_yaml['Total_video_checkpoint'] + args.num_videos
    main_yaml['Total_iterations'] = main_yaml['Total_iterations'] + 1

    rows = read_data(base_data_dir + args.csv_file_name, start_index=total_number_of_videos, end_index=args.num_videos + total_number_of_videos)
    video_urls = []

    for index in rows:
        video_urls.append(index[1])

    indexes_to_delete = download_videos(video_urls, base_directory)

    # Save the relevant data to iteration YAML file
    track_file = f'../logs/logs_{args.csv_file_name.replace(".csv", "")}/track_{main_yaml["Total_iterations"]}.yaml'
    iter_data = createOrLoadYamlFile(track_file)
    iter_data['Source'] = args.csv_file_name
    iter_data['failed_indexes'] = indexes_to_delete
    iter_data['video_checkpoint_from_to'] = [total_number_of_videos, args.num_videos + total_number_of_videos]
    iter_data['total_videos'] = args.num_videos

if __name__ == "__main__":
    main()
    