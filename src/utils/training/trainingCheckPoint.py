import yaml
import sys
import os

sys.path.append('../../common')
import paths

def getLatestTrainingCheckpoint(csv_file_path):
    # Load the YAML content from the file into a Python dictionary
    with open(csv_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the data in the Python dictionary
    return data["Total_iterations"]

def getTrainValTxtRange(csv_file:str, last_training_checkpoint:int):
    file_dir_name = os.path.join(paths.base_download_checkpoint_dir, f"logs_{csv_file.replace('.csv', '')}")
    file_name = f"track_{last_training_checkpoint}.yaml"

    # Load Train Txt
    with open(os.path.join(file_dir_name, file_name), 'r') as train_file:
        train_data = yaml.safe_load(train_file)

    # Load Validation Txt
    file_dir_name = os.path.join(paths.base_download_checkpoint_dir, f"logs_{csv_file.replace('train.csv', 'val')}")
    # Assuming we only have one validation Tensor
    file_name = "track_0.yaml"
    with open(os.path.join(file_dir_name, file_name), 'r') as val_file:
        val_data = yaml.safe_load(val_file)


    return train_data['video_checkpoint_from_to'], val_data['video_checkpoint_from_to']
