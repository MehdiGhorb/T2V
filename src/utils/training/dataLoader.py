import os
import sys

sys.path.append('../../common')
import paths
sys.path.append(paths.tensor_gen_path)
from saveTensor import loadTensor
from trainingCheckPoint import getLatestTrainingCheckpoint, getTrainValTxtRange
sys.path.append(os.path.join(paths.utils_dir, 'video_preprocessing'))
from videoPrepHelper import read_data

def loadTrainValTxt(csv_file_path):

    '''Variables'''
    training_yaml = csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    last_checkpoint = getLatestTrainingCheckpoint(os.path.join(paths.training_checkpoint_dir + f'/{csv_file_path.replace("_train.csv", "")}', training_yaml))
    train_txt_indexes, val_txt_indexes = getTrainValTxtRange(csv_file=csv_file_path, 
                                                            last_training_checkpoint=last_checkpoint)
    
    '''Read Training Text Data'''
    training_rows = read_data(os.path.join(paths.base_data_dir + '/csv_files/customised', csv_file_path),
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
    validation_rows = read_data(os.path.join(paths.base_data_dir + '/csv_files/customised', csv_file_path.replace("_train", "_val")),
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
    training_yaml = csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    last_checkpoint = getLatestTrainingCheckpoint(os.path.join(paths.training_checkpoint_dir + f'/{csv_file_path.replace("_train.csv", "")}', training_yaml))
    FolderPath = f'/{csv_file_path.replace(".csv", "")}'
    train_tensor = loadTensor(os.path.join(paths.tensor_path + FolderPath, f'track_{last_checkpoint}.pt'))
    print("\nTraining Tensor has been loaded successfully!!\n")

    return train_tensor

def loadValTensor(csv_file_path):
    '''Load Validation Tensor'''
    # Assuming there is only one Tensor for Validation (If running out of Memory, simply divide the Tensor into smaller Tensors and randomly choose one of them)
    FolderPath = f'/{csv_file_path.replace("_train.csv", "_val")}'
    val_tensor = loadTensor(os.path.join(paths.tensor_path + FolderPath, 'track_0.pt'))
    print("\nValidation Tensor has been loaded successfully!!\n")

    return val_tensor
