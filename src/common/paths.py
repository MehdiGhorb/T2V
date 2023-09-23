import os

# Base directories
base_dir = '/mnt/c/Users/ghorb/OneDrive/Desktop/T2V'
base_data_dir = os.path.join(base_dir, 'data')
base_src_dir = os.path.join(base_dir, 'src')

# Subdirectories
base_mp4video_directory = os.path.join(base_data_dir, 'videos', 'original_videos')
final_gif_directory = os.path.join(base_data_dir, 'videos', 'gifs')
base_download_checkpoint_dir = os.path.join(base_data_dir, 'download_checkpoint')
helper_path = os.path.join(base_src_dir, 'helper')
tensor_path = os.path.join(base_data_dir, 'tensors')
downloader_path = os.path.join(base_src_dir, 'video_preprocessing')
tensor_gen_path = os.path.join(base_src_dir, 'tensorMaker')
training_checkpoint_dir = os.path.join(base_data_dir, 'training_checkpoint')