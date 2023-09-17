'''
Convert videos to GIFs and create Tensors of them
Run this in Shell or use Jupyter notebook:

'''

import argparse
from tqdm import trange, tqdm
from yamlEditor import *
import sys
sys.path.append('../helper')
from dataLoader import *
from videoAnnotation import *
sys.path.append('../common')
import paths
import warnings

def main():
    #parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    #parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    #parser.add_argument('num_videos', type=int, help='Number of videos per iteration')
    #args = parser.parse_args()

    # create an empty dictionary to store tensors
    gif_dict = {}
    # gif index
    gif_index = 0
    # List of converted tensors
    tensor_list = []

    # get all the video names
    videos = getVideoNames(paths.base_mp4video_directory)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(videos, desc="Processing videos"):
            cropped_video = crop_video_frames(i, 30, 30, 30, 30)
            resized_video = resize_frames(cropped_video, 64, 64)
            reduced_length = reduce_frames(resized_video, 5, 10)
            create_gif(reduced_length, f"/content/gifs/Gif_{gif_index}.gif")

            # create the dictionary of tensors
            gif_name = f"gif_{gif_index}"
            gif_value = f"/content/gifs/Gif_{gif_index}.gif"
            gif_dict[gif_name] = gif_value
            gif_index += 1

    for gif in gif_dict.values():
        tensor_list.append(gif_to_tensor(gif))

    desired_frame_number = 10  # Replace with your desired frame number

    # Iterate through the tensors and check their frame numbers
    for i, tensor in enumerate(tensor_list):

        if tensor.shape != sample_shape:
            # If the frame number is not the desired one, append its index to the list
            indexes_to_delete.append(i)

    # Now, let's delete the undesired tensors from the list
    # We'll iterate through the list of indices to delete and remove those tensors
    for index in reversed(indexes_to_delete):
        del tensor_list[index]
    print(indexes_to_delete)

    # At this point, the undesired tensors have been removed from the 'tensors' list,
    # and 'indices_to_delete' contains the indices of the removed tensors.
