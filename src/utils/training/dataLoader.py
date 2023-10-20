import os
import sys

sys.path.append('../../common')
import paths

sys.path.append(paths.TENSOR_GEN_PATH)
from saveTensor import loadTensor
from saveTensor import loadTensor
from trainingCheckPoint import getLatestTrainingCheckpoint, getTrainValTxtRange

sys.path.append(os.path.join(paths.UTILS_DIR, 'video_preprocessing'))
from videoPrepHelper import read_data

sys.path.append(paths.CLOUD_UTILS)
from cloudUtils import downloadFile, getFolderIDByName, checkForUpdates


def loadTrainValTxt(csv_file_path):

    '''Variables'''
    training_yaml = csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    last_checkpoint = getLatestTrainingCheckpoint(os.path.join(paths.TRAINING_CHECKPOINT_DIR + f'/{csv_file_path.replace("_train.csv", "")}', training_yaml))
    train_txt_indexes, val_txt_indexes = getTrainValTxtRange(csv_file=csv_file_path, 
                                                             last_training_checkpoint=last_checkpoint)
    
    '''Read Training Text Data'''
    training_rows = read_data(os.path.join(paths.BASE_DATA_DIR + '/csv_files/customised', csv_file_path),
                              start_index=train_txt_indexes[0]-2,
                              end_index=train_txt_indexes[1]-2)
    train_text = []
    # Read video descriptions
    for index in training_rows:
        train_text.append(index[1])
    # Delete for memory efficiency purposes
    del training_rows
    # garbage collector (Test if it really helps improve memory efficiency)

    '''Read Validation Text Data'''
    validation_rows = read_data(os.path.join(paths.BASE_DATA_DIR + '/csv_files/customised', csv_file_path.replace("_train", "_val")),
                                start_index=val_txt_indexes[0]-2,
                                end_index=val_txt_indexes[1]-2)
    val_text = []
    # Read video descriptions
    for index in validation_rows:
        val_text.append(index[1])
    # Delete for memory efficiency purposes
    del validation_rows

    return train_text, val_text

def loadTrainingTensor(csv_file_path):
    '''Load Training Tensor'''

    '''Load/Download track YAML'''
    # training_yaml is the file that keeps tracks of training iteration and track_yaml is the file that contains tensor metadata
    training_yaml = csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    training_yaml_dir_name = f'/{csv_file_path.replace("_train.csv", "")}'
    training_yaml_path = paths.TRAINING_CHECKPOINT_DIR + training_yaml_dir_name
    last_checkpoint = getLatestTrainingCheckpoint(os.path.join(training_yaml_path, training_yaml))
    track_yaml = f'track_{last_checkpoint}.yaml'
    download_yaml_path = paths.BASE_DOWNLOAD_CHECKPOINT_DIR + "/logs_" + f"{csv_file_path.replace('.csv', '')}"

    # Check if the track_yaml exists
    if not os.path.exists(os.path.join(download_yaml_path, track_yaml)):
        # If not check if cloud has this file
        yaml_folder_ID = getFolderIDByName("logs_" + f"{csv_file_path.replace('.csv', '')}")
        yaml_download_ID = checkForUpdates(file_name=track_yaml, 
                                           folder_id=yaml_folder_ID)
        if not yaml_download_ID:
            sys.exit(f"No new training tensors ({track_yaml}), please add more tensors!")
        else:
            downloadFile(file_n=track_yaml,
                         file_id=yaml_download_ID,
                         save_path=download_yaml_path)
            print(f'Added {track_yaml} to {download_yaml_path} .\n')
    
    '''Load/Downlaod Tensor'''
    # The process is the same as loading/downloading track_yaml file
    tensor_name = f'track_{last_checkpoint}.pt'
    training_tensor_dir_name = f'/{csv_file_path.replace(".csv", "")}'
    #training_tensor_path = os.path.join(paths.TENSOR_PATH, training_tensor_dir_name) for any reason this line doesn't really concatenate the directories
    training_tensor_path = paths.TENSOR_PATH + training_tensor_dir_name
    full_tensor_path = os.path.join(training_tensor_path, tensor_name)

    if not os.path.exists(full_tensor_path):
        tensor_folder_ID = getFolderIDByName(f'{csv_file_path.replace(".csv", "")}')
        tensor_download_ID = checkForUpdates(file_name=tensor_name, 
                                             folder_id=tensor_folder_ID)
        if not tensor_download_ID:
            sys.exit(f"No new Training Tensors {tensor_name}, Please add more Tensors!")
        else:
            downloadFile(file_n=tensor_name,
                         file_id=tensor_download_ID,
                         save_path=training_tensor_path)
            print(f'Added {tensor_name} to {training_tensor_path} .\n')

    # Load the Tensor
    train_tensor = loadTensor(os.path.join(paths.TENSOR_PATH + training_tensor_dir_name, f'track_{last_checkpoint}.pt'))
    print(f"\nLoaded training tensor successfully (track_{last_checkpoint}.pt)\n")

    return train_tensor

def loadValTensor(csv_file_path):
    '''Load Validation Tensor'''
    # Assuming there is only one Tensor for Validation (If running out of Memory, simply divide the Tensor into smaller Tensors and randomly choose one of them)
    FolderPath = f'/{csv_file_path.replace("_train.csv", "_val")}'
    val_tensor = loadTensor(os.path.join(paths.TENSOR_PATH + FolderPath, 'track_0.pt'))
    print("\nValidation tensor loaded successfully!!\n")

    return val_tensor
