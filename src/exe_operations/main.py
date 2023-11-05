'''!!Use Labs instead!!'''
import argparse
import os
import sys
import ast
from tqdm import trange
import torch
import yaml
from torch.optim.lr_scheduler import StepLR
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# Import local modules
# TODO: This way of importing is suboptimal
sys.path.append('../common')
import paths

sys.path.append(os.path.join(paths.UTILS_DIR, 'training'))
from dataLoader import *
from earlyStopping import EarlyStopping

sys.path.append(paths.CLOUD_UTILS)
from cloudUtils import uploadTensor, getFolderIDByName, mainModelCloudBackupControl, mainModelLocalBackupControl

'''Load training parameters'''
with open(os.path.join(paths.CONFIG_DIR, 'trainingParams.yaml'), 'r') as f:
    training_params = yaml.safe_load(f)

'''Load Tensor parameters'''
with open(os.path.join(paths.CONFIG_DIR, 'tensorConfig.yaml'), 'r') as f:
    tensor_params = yaml.safe_load(f)

'''Fodler IDs'''
with open(os.path.join(paths.CLOUD_IDS, 'folder_ids.yaml'), 'r') as f:
    f_ids = yaml.safe_load(f)

'''Training parameters'''
NUM_ITERATIONS = int(training_params['num_iterations'])
TIME_STEPS = int(training_params['time_steps'])
BATCH_SIZE = int(training_params['batch_size'])
CHECKPOINT_INTERVAL = int(training_params['checkpoint_interval'])
DATASET_SIZE = int(training_params['dataset_size'])
TRAINING_LOSS_TYPE = training_params['training_loss_type']
UNET_DIM = training_params['Unet'][0]['dim']
TXT_COND = training_params['Unet'][1]['use_text_cond']
DIM_MULTS = ast.literal_eval(training_params['Unet'][2]['dim_mults'])

'''Tensor Parameters'''
IMAGE_SIZE = int(tensor_params['frame_size']/2)
NUM_FRAMES = int(tensor_params['frame_num'])

def main():
    parser = argparse.ArgumentParser(description='Main Training Script')
    parser.add_argument('csv_file_path', help='Path to the CSV file containing video descriptions')
    args = parser.parse_args()

    '''Main Model Cloud Folder ID'''
    MODEL_FOLDER_ID = getFolderIDByName(f'{args.csv_file_path}'.replace("train.csv", "models"))

    '''Read Training Text Data, Load Training/Validation Tensor'''
    train_tensor = loadTrainingTensor(args.csv_file_path)
    val_tensor = loadValTensor(args.csv_file_path)
    # loadTrainValTxt() must be after loadTrainingTensor() function, since new tensors are only downloaded in loadTrainingTensor() function
    train_text, val_text = loadTrainValTxt(args.csv_file_path)

    '''Model'''
    model = Unet3D(
        dim=UNET_DIM,
        use_bert_text_cond=TXT_COND,  # this must be set to True to auto-use the bert model dimensions
        dim_mults=DIM_MULTS,
    )

    # Print out the number of paramers
    total_params = sum(p.numel() for p in model.parameters())
    memory_usage_gb = (total_params * 4) / (1024**3)
    print(f"Total parameters in the model: {total_params}")
    print(f"Memory usage: {memory_usage_gb:.2f} GB")


    '''Load the latest model'''
    TRAINING_YAML = args.csv_file_path.replace("customised", "training").replace("_train.csv", ".yaml")
    last_checkpoint = getLatestTrainingCheckpoint(os.path.join(paths.TRAINING_CHECKPOINT_DIR + f'/{args.csv_file_path.replace("_train.csv", "")}', TRAINING_YAML))
    model_path = os.path.join(paths.MODEL_DIR, f'{args.csv_file_path}'.replace('.csv', ''))
    CHECKPOINT_PATH = os.path.join(model_path, f'main_model_{last_checkpoint}.pth')

    '''Early Stopping instance'''
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.001, path=os.path.join(paths.MODEL_DIR, f'main_model_{last_checkpoint}.pth'))

    if last_checkpoint != 0:

        # Load the checkpoint
        try:
            checkpoint = torch.load(os.path.join(model_path, f'main_model_{last_checkpoint-1}.pth'))
            print(f"Main Model ( main_model_{last_checkpoint-1}.pth ) loaded successfully\n")
        except Exception as e:
            raise Exception(f"Failed to load checkpoint: {str(e)}")
        # Separate keys for model and optimizer
        model_state_dict = checkpoint['model_state_dict']
        # Check if the model state_dict is empty (indicating loading failure)
        if not model_state_dict:
            raise Exception("Failed to load the model state_dict.\nRuntime interuppted.")
        # Load the model state_dict
        model.load_state_dict(model_state_dict)
        # Optional: Load other training-related information if needed
        iteration = checkpoint['iteration']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']
        # Set the model in training mode
        model.train()

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,          # height and width of frames
        num_frames=NUM_FRAMES,          # number of video frames
        timesteps=TIME_STEPS,           # number of steps
        loss_type=TRAINING_LOSS_TYPE    # L1 or L2
    )

    # Assuming you have your optimizer defined
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Define a learning rate scheduler (dynamic learning rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Save the new model
    torch.save({
                'iteration': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0,
                'val_loss': 0,
        }, CHECKPOINT_PATH)

    for iteration in trange(NUM_ITERATIONS):

        # Sample indices for the current batch
        batch_indices = torch.randint(0, DATASET_SIZE, (BATCH_SIZE,))
        
        # Sample a batch of training data
        batch_videos = train_tensor[batch_indices]
        batch_text = [train_text[idx] for idx in batch_indices]

        # Forward pass with text conditioning
        loss = diffusion(batch_videos, cond=batch_text)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        print(f"Iteration [{iteration}]: Loss = {loss.item()}")
        if iteration % 100 == 0:
            print(f"Train Iteration [{iteration}/{NUM_ITERATIONS}]: Loss = {loss.item()}")
            
        # Validation every checkpoint_interval iterations
        if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
            val_loss = 0.0
            num_val_batches = len(val_tensor) // BATCH_SIZE

            for val_batch_start in range(0, len(val_tensor), BATCH_SIZE):
                val_batch_videos = val_tensor[val_batch_start:val_batch_start + BATCH_SIZE]
                val_batch_text = val_text[val_batch_start:val_batch_start + BATCH_SIZE]

                with torch.no_grad():
                    val_batch_loss = diffusion(val_batch_videos, cond=val_batch_text)
                    val_loss += val_batch_loss.item()

            val_loss /= num_val_batches
            print(f"Validation Iteration [{iteration}/{NUM_ITERATIONS}]: Loss = {val_loss}")

            # Use the EarlyStopping object to track validation loss
            early_stopping(val_loss, model)

            # Check if early stopping criteria are met
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break  # Exit the training loop
            
            # Save checkpoint
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss,
            }, CHECKPOINT_PATH)
            print(f"Checkpoint saved at iteration {iteration + 1}")

    # Update training Checkpoint
    with open(os.path.join(paths.TRAINING_CHECKPOINT_DIR + f'/{args.csv_file_path.replace("_train.csv", "")}', TRAINING_YAML), "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    yaml_data["Total_iterations"] = yaml_data["Total_iterations"] + 1
    with open(os.path.join(paths.TRAINING_CHECKPOINT_DIR + f'/{args.csv_file_path.replace("_train.csv", "")}', TRAINING_YAML), "w") as f:
        yaml.dump(yaml_data, f)

    # Upload the model to the cloud
    _=uploadTensor(file_path=os.path.join(model_path, f'main_model_{last_checkpoint}.pth'),
                   folder_id=MODEL_FOLDER_ID,
                   file_name_to_upload=f'main_model_{last_checkpoint}.pth')
    # Make sure there are no more than 5 models on the cloud to reduce storage usage
    mainModelCloudBackupControl(MODEL_FOLDER_ID)
    mainModelLocalBackupControl(model_path)

    print(" \n'''Training finished'''\n ")

if __name__ == "__main__":
    main()
    