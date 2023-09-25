import os
from trainingCheckPoint import getLatestCheckpoint
from saveTensor import loadTensor
from dataLoader import read_data
from common import paths

def loadTrainValTxt(args):
    '''Read Training Text Data'''
    training_rows = read_data(os.path.join(paths.base_data_dir + '/csv_files/customised', args.csv_file_path))
    train_text = []
    # Read video descriptions
    for index in training_rows:
        train_text.append(index[1])
    # Delete for memory efficiency purposes
    del training_rows
    # garbage collector (Test if it really helps improve memory efficiency)

    '''Read Validation Text Data'''
    validation_rows = read_data(os.path.join(paths.base_data_dir + '/csv_files/customised', args.csv_file_path.replace("_train", "_val")))
    val_text = []
    # Read video descriptions
    for index in validation_rows:
        val_text.append(index[1])
    # Delete for memory efficiency purposes
    del validation_rows

    return train_text, val_text

def loadTrainingTensor(args):

    '''Load Training Tensor'''
    training_yaml = args.csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    last_checkpoint = getLatestCheckpoint(os.path.join(paths.training_checkpoint_dir + f'/{args.csv_file_path.replace("_train.csv", "")}', training_yaml))
    FolderPath = f'/{args.csv_file_path.replace(".csv", "")}'
    train_tensor = loadTensor(os.path.join(paths.tensor_path + FolderPath, f'track_{last_checkpoint}.pt'))
    print("\nTraining Tensor has been loaded successfully!!\n")

    return train_tensor

def loadValTensor(args):

    '''Load Validation Tensor'''
    # Assuming there is only one Tensor for Validation (If running out of Memory, simply divide the Tensor into smaller Tensors and randomly choose one of them)
    FolderPath = f'/{args.csv_file_path.replace("_train.csv", "_val")}'
    val_tensor = loadTensor(os.path.join(paths.tensor_path + FolderPath, 'track_0.pt'))
    print("\nValidation Tensor has been loaded successfully!!\n")

    return val_tensor
