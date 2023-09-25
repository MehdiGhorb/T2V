'''
To use this script run python3 customizeCSV.py {csvfile.csv}
'''

from utils.video_preocessing.videoPrepHelper import createCustomedVideoCsvFile, read_data
import argparse
from tqdm import tqdm
import sys
sys.path.append('../common')
import paths
import os

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    args = parser.parse_args()

    rows = read_data(os.path.join(paths.base_data_dir + "/csv_files/original", args.csv_file_name), start_index=0, end_index=11000000)

    video_urls = []
    video_desc = []

    for index in tqdm(rows, desc="Customizing CSV File ..."):
        video_urls.append(index[1])
        # Remove line breaks and newline characters from descriptions
        description = index[4].replace("\n", " ").replace("\r", " ")
        video_desc.append(description)

    '''Create customised CSV file'''
    # Format: {Video URLs, Desc}
    createCustomedVideoCsvFile(video_urls=video_urls, 
                               video_names=video_desc, 
                               csv_filename=os.path.join(paths.base_data_dir + "/csv_files/customised", f"customised_{args.csv_file_name}"))

if __name__ == "__main__":
    main()
