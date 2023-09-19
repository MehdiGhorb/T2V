'''
Convert videos to GIFs and create Tensors of them
Run this in Shell or use Jupyter notebook:

'''

import torch
import argparse
from tqdm import trange, tqdm
from yamlEditor import *
import sys
sys.path.append('../helper')
from dataLoader import getVideoNames, updateIterationYamlFile
from videoAnnotation import *
sys.path.append('../common')
import paths
sys.path.append('../video_preprocessing')
from yamlEditor import loadMainYamlFile
from saveTensor import saveBuffer
import warnings
import yaml

def main():        
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    args = parser.parse_args()

    main_yaml = loadMainYamlFile(paths.base_download_checkpoint_dir + "main_customised/" + f"track_{args.csv_file_name.replace('.csv', '')}.yaml")
    video_yamlFile = paths.base_download_checkpoint_dir + f'logs_{args.csv_file_name.replace(".csv", "")}/track_{main_yaml["Total_iterations"]-1}.yaml'
    # Create an empty dictionary to store tensors
    gif_dict = {}
    # gif index
    gif_index = 0
    # List of converted tensors
    tensor_list = []
    # Indexes to delete (unqualified tensors)
    indexes_to_delete = []
    # get all the video names
    videos = getVideoNames(video_yamlFile)

    # Read the config file
    with open("config_values/config.yaml", "r") as config:
        tensor_config = yaml.load(config, Loader=yaml.FullLoader)

    # Create a sample tensor to compare and remove the unqualified tensors
    size = (tensor_config["channels"], 
            tensor_config["frame_num"], 
            tensor_config["frame_size"]/2, 
            tensor_config["frame_size"]/2)
    sample_shape = torch.tensor(size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for video in tqdm(videos, desc="Processing videos..."):
            # Crop videos
            cropped_video = crop_video_frames(video, 
                                              tensor_config["crop_size"], 
                                              tensor_config["crop_size"], 
                                              tensor_config["crop_size"], 
                                              tensor_config["crop_size"])
            # Resize the frames
            resized_video = resize_frames(cropped_video, tensor_config["frame_size"], 
                                          tensor_config["frame_size"])
            # Reduce the length
            reduced_length = reduce_frames(resized_video, tensor_config["length"], 
                                           tensor_config["frame_num"])
            # Convert the videos to GIFs
            create_gif(reduced_length, f"/content/gifs/Gif_{gif_index}.gif")

            # create the dictionary of tensors
            gif_name = f"gif_{gif_index}"
            gif_value = f"/content/gifs/Gif_{gif_index}.gif"
            gif_dict[gif_name] = gif_value
            gif_index += 1

    for gif in tqdm(gif_dict.values(), desc="Converting GIFs to Tensors... "):
        tensor_list.append(gif_to_tensor(gif))

    # Iterate through the tensors and check their frame numbers
    for i, tensor in enumerate(tensor_list):

        if tensor.shape != sample_shape:
            # If the frame number is not the desired one, append its index to the list
            indexes_to_delete.append(i)

    # Delete the undesired tensors from the list (if they exist)
    if len(indexes_to_delete) != 0:
        for index in reversed(indexes_to_delete):
            del tensor_list[index]

        # Update the tensor YAML file accordingly
        with open(video_yamlFile, "r") as temp:
            tensor_info = yaml.load(temp, Loader=yaml.FullLoader)
        source = tensor_info["Source"]
        failed_indexes = tensor_info["failed_indexes"]
        total_videos = tensor_info["total_videos"]
        video_checkpoint_range = tensor_info["video_checkpoint_from_to"]

        for index in indexes_to_delete:
            failed_indexes.append(index)

        updateIterationYamlFile(video_yamlFile, 
                                source=source,
                                failed_indexes=failed_indexes,
                                total_videos=total_videos,
                                video_checkpoint_from_to=video_checkpoint_range)
    
    # Save the tensor
    saveBuffer(tensor_list=tensor_list,
               tensor_path=paths.tensor_path + f'track_{main_yaml["Total_iterations"]-1}.pt')
    