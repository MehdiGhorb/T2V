import subprocess
import argparse
import sys
sys.path.append('../common')
import paths

# Download Videos
downloader_path = paths.downloader_path
videoDownloader = "videoDownloader.py"
# Generate tensors and remove the videos
tensor_gen_path = paths.tensor_gen_path
tensorGenerator = "tensorGenerator.py"

def main():
    parser = argparse.ArgumentParser(description='Download videos and convert them to tensors (Main execution script)')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    parser.add_argument('num_videos', type=int, help='Number of videos per iteration')
    parser.add_argument('iterations', type=int, help='Number of iterations')
    args = parser.parse_args()

    for _ in range(args.iterations):
        # Download Videos
        subprocess.run(["python3", videoDownloader, args.csv_file_name, str(args.num_videos)], cwd=downloader_path)
        # Create Tensors and remove videos
        subprocess.run(["python3", tensorGenerator, args.csv_file_name], cwd=tensor_gen_path)

if __name__ == "__main__":
    main()