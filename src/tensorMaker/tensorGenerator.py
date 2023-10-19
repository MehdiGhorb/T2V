'''
Convert videos to GIFs and create Tensors of them
Run this in Shell or use Jupyter notebook:

'''
import torch
import argparse
from tqdm import tqdm
import sys
import warnings
import yaml
import os

sys.path.append('../utils/video_preprocessing')
from videoPrepHelper import getVideoNames, removeDirContent
from videoAnnotation import *

sys.path.append('../common')
import paths

sys.path.append('../video_preprocessing')
from yamlEditor import loadMainYamlFile, updateIterationYamlFile
from saveTensor import saveTensor

sys.path.append(paths.CLOUD_UTILS)
from cloudUtils import uploadYAML, getFolderIDByName, uploadTensor

def main():        
    parser = argparse.ArgumentParser(description='Download videos from CSV URLs')
    parser.add_argument('csv_file_name', help='Path to the CSV file containing video URLs')
    args = parser.parse_args()

    main_yaml = loadMainYamlFile(os.path.join(paths.BASE_DOWNLOAD_CHECKPOINT_DIR + "/main_customised", f"track_{args.csv_file_name.replace('.csv', '')}.yaml"))
    video_yamlFile = os.path.join(paths.BASE_DOWNLOAD_CHECKPOINT_DIR + f'/logs_{args.csv_file_name.replace(".csv", "")}', f'track_{main_yaml["Total_iterations"]-1}.yaml')
    # Create an empty dictionary to store tensors
    gif_dict = {}
    # gif index
    gif_index = 0
    # List of converted tensors
    tensor_list = []
    # Indexes to delete (unqualified tensors)
    indexes_to_delete = []
    # get all the video names
    videos = getVideoNames(paths.BASE_MP4VIDEO_DIRECTORY)

    # Read the config file
    with open(os.path.join(paths.CONFIG_DIR, 'tensorConfig.yaml'), "r") as config:
        tensor_config = yaml.load(config, Loader=yaml.FullLoader)

    # Create a sample tensor to compare and remove the unqualified tensors
    size = (int(tensor_config["channels"]), 
            int(tensor_config["frame_num"]), 
            int(tensor_config["frame_size"]/2), 
            int(tensor_config["frame_size"]/2))
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
            resized_video = resize_frames(cropped_video, 
                                          tensor_config["frame_size"], 
                                          tensor_config["frame_size"])
            # Reduce the length
            reduced_length = reduce_frames(resized_video, 
                                           tensor_config["length"], 
                                           tensor_config["frame_num"])
            # Convert the videos to GIFs
            create_gif(reduced_length, paths.FINAL_GIF_DIRECTORY + f"/Gif_{gif_index}.gif")

            # create the dictionary of tensors
            gif_name = f"gif_{gif_index}"
            gif_value = paths.FINAL_GIF_DIRECTORY + f"/Gif_{gif_index}.gif"
            gif_dict[gif_name] = gif_value
            gif_index += 1

    # Remove mp4 videos
    removeDirContent(paths.BASE_MP4VIDEO_DIRECTORY)

    # Convert GIFs to Tensors
    for gif in tqdm(gif_dict.values(), desc="Converting GIFs to Tensors... "):
        tensor_list.append(gif_to_tensor(gif))

    # Iterate through the tensors and check their frame numbers
    for i, tensor in enumerate(tensor_list):

        if torch.equal(tensor, sample_shape):
            # If the frame number is not the desired one, append its index to the list
            indexes_to_delete.append(i)

    # Delete the unqualified tensors from the list (if they exist)
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
    saveTensor(tensor_list=tensor_list,
               tensor_path=os.path.join(paths.TENSOR_PATH + f'/{args.csv_file_name.replace(".csv", "")}', f'track_{main_yaml["Total_iterations"]-1}.pt'))

    # Upload Tensor
    folder_id = getFolderIDByName(name=f'{args.csv_file_name.replace(".csv", "")}')
    tensor_ID = uploadTensor(file_path=os.path.join(paths.TENSOR_PATH + f'/{args.csv_file_name.replace(".csv", "")}', f'track_{main_yaml["Total_iterations"]-1}.pt'), 
                             folder_id=folder_id,
                             file_name_to_upload=f'track_{main_yaml["Total_iterations"]-1}.pt'
                            )

    folder_id = getFolderIDByName(name=f'logs_{args.csv_file_name.replace(".csv", "")}')
    uploadYAML(file_path=video_yamlFile, 
                    folder_id=folder_id,
                    file_name_to_upload=f'track_{main_yaml["Total_iterations"]-1}.yaml',
                    tensor_id=tensor_ID)
    
    # Remove GIFs
    removeDirContent(paths.FINAL_GIF_DIRECTORY)

    print("\nTensor saved successfully!\n")

if __name__ == "__main__":
    main()
    