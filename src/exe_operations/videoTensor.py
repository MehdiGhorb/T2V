import subprocess
import argparse
import os
import sys
sys.path.append('../common')
import paths

sys.path.append(paths.CLOUD_UTILS)
from cloudUtils import authenticate

# Check token expiry status
is_expired = authenticate(os.path.join(paths.CLOUD_CREDS, 'token.json'))
if is_expired == True:
    sys.exit('Google Drive token has expired, Please renew the token\n')

# Download Videos
downloader_path = paths.DOWNLOADER_PATH
videoDownloader = "videoDownloader.py"
# Generate tensors and remove the videos
tensor_gen_path = paths.TENSOR_GEN_PATH
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
