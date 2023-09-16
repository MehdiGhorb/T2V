'''
To use this script run python3 customizeCSV.py {csvfile.csv}
'''

from dataLoader import createCustomedVideoCsvFile, read_data
import argparse
from tqdm import tqdm
import sys
sys.path.append('../common')
import paths

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    args = parser.parse_args()

    rows = read_data(paths.base_data_dir + "original/" + args.csv_file_name, start_index=0, end_index=11000000)
    #parser.add_argument('num_videos', type=int, help='Number of videos per iteration')
    video_urls = []
    video_desc = []

    for index in tqdm(rows, desc="Customizing CSV File ..."):
        video_urls.append(index[1])
        video_desc.append(index[4])

    # Create customed CSV file
    # Format: {Video URLs, Desc}
    createCustomedVideoCsvFile(video_urls=video_urls, video_names=video_desc, csv_filename=paths.base_data_dir + "customed/" + f"customed_{args.csv_file_name}")

if __name__ == "__main__":
    main()

