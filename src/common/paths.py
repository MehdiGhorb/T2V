import os

# Base directories
# TODO base directory to be calculated rather than hard-coding
BASE_DIR = '/mnt/c/Users/ghorb/OneDrive/Desktop/T2V'
BASE_DATA_DIR = os.path.join(BASE_DIR, 'data')
BASE_SRC_DIR = os.path.join(BASE_DIR, 'src')
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')

# Subdirectories
BASE_MP4VIDEO_DIRECTORY = os.path.join(BASE_DATA_DIR, 'videos', 'original_videos')
FINAL_GIF_DIRECTORY = os.path.join(BASE_DATA_DIR, 'videos', 'gifs')
BASE_DOWNLOAD_CHECKPOINT_DIR = os.path.join(BASE_DATA_DIR, 'download_checkpoint')
HELPER_PATH = os.path.join(BASE_SRC_DIR, 'helper')
TENSOR_PATH = os.path.join(BASE_DATA_DIR, 'tensors')
DOWNLOADER_PATH = os.path.join(BASE_SRC_DIR, 'video_preprocessing')
TENSOR_GEN_PATH = os.path.join(BASE_SRC_DIR, 'tensorMaker')
TRAINING_CHECKPOINT_DIR = os.path.join(BASE_DATA_DIR, 'training_checkpoint')
MODEL_DIR = os.path.join(BASE_DATA_DIR, 'models')
MODEL_BACKUP_DIR = os.path.join(BASE_DATA_DIR, 'models', 'backups')
COMMON_DIR = os.path.join(BASE_SRC_DIR, 'common')
UTILS_DIR = os.path.join(BASE_SRC_DIR, 'utils')
CLOUD_CREDS = os.path.join(BASE_DATA_DIR, 'cloud_credentials')
