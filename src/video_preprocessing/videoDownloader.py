'''
Download the videos by providing the csv path to the URLs and a path to save the videos
Run this in Shell or use Jupyter notebook:
python3 your_script.py /path/to/your.csv /path/to/save/videos 10
'''
import argparse
from yamlEditor import *
import sys

sys.path.append('../utils/video_preprocessing')
from videoPrepHelper import *
sys.path.append('../common')
import paths

def main():
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    parser.add_argument('num_videos', type=int, help='Number of videos per iteration')
    args = parser.parse_args()

    # Read data from main YAML
    main_yaml = loadMainYamlFile(os.path.join(paths.BASE_DOWNLOAD_CHECKPOINT_DIR + "/main_customised", f"track_{args.csv_file_name.replace('.csv', '')}.yaml"))
    total_number_of_videos = main_yaml['Total_video_checkpoint']
    number_of_iterations = main_yaml['Total_iterations']

    rows = read_data(os.path.join(paths.BASE_DATA_DIR + "/csv_files/customised", args.csv_file_name), 
                     start_index=total_number_of_videos, 
                     end_index=args.num_videos + total_number_of_videos-1)
    video_urls = []

    for index in rows:
        video_urls.append(index[0])

    try:
        indexes_to_delete = download_videos(video_urls, paths.BASE_MP4VIDEO_DIRECTORY)
    except:
        # Do not Modify any checkpoint information if the download process was unsuccessful
        sys.exit()

    # Save the relevant data to iteration YAML file
    track_file = os.path.join(paths.BASE_DOWNLOAD_CHECKPOINT_DIR + f'/logs_{args.csv_file_name.replace(".csv", "")}', f'track_{main_yaml["Total_iterations"]}.yaml')
    total_vids = countVideosInDirectory(paths.BASE_MP4VIDEO_DIRECTORY)
    _ = createOrLoadYamlFile(track_file)
    updateIterationYamlFile(track_file,
                            source=args.csv_file_name,
                            failed_indexes=indexes_to_delete,
                            total_videos=total_vids,
                            video_checkpoint_from_to=[total_number_of_videos+2, args.num_videos + total_number_of_videos+1])

    # Update main YAML file
    main_yaml_file = os.path.join(paths.BASE_DOWNLOAD_CHECKPOINT_DIR + "/main_customised", f"track_{args.csv_file_name.replace('.csv', '')}.yaml")
    updateMainYamlFile(main_yaml_file,
                       source=args.csv_file_name.replace('.csv', ''),
                       total_iterations=main_yaml['Total_iterations'] + 1,
                       total_video_check_point=main_yaml['Total_video_checkpoint'] + total_vids)

if __name__ == "__main__":
    main()
    