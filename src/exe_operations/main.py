'''!!Use Labs instead!!'''
import argparse
import os
import sys
from tqdm import trange
import torch
import yaml

from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# Import local modules
sys.path.append('../common')
import paths

# TODO: This way of importing is suboptimal
sys.path.append('../utils/training')
from dataLoader import *

with open(os.path.join(paths.config_dir, 'trainingParams.yaml'), 'r') as f:
    params = yaml.safe_load(f)

# Use the parameters in your training code
num_iterations = params['num_iterations']
batch_size = params['batch_size']
checkpoint_interval = params['checkpoint_interval']
dataset_size = params['dataset_size']

def main():
    parser = argparse.ArgumentParser(description='Main Training Script')
    parser.add_argument('csv_file_path', help='Path to the CSV file containing video descriptions')
    args = parser.parse_args()

    '''Read Training Text Data'''
    train_text, val_text = loadTrainValTxt(args)

    '''Load Training Tensor'''
    train_tensor = loadTrainingTensor(args)

    '''Load Validation Tensor'''
    val_tensor = loadValTensor(args)

    '''Model'''
    model = Unet3D(
        dim=64,
        use_bert_text_cond=True,  # this must be set to True to auto-use the bert model dimensions
        dim_mults=(1, 2, 4, 8),
    )

    # Load the latest model
    if last_checkpoint != 0: 
        model.load_state_dict(torch.load(os.path.join(paths.model_dir, 'main_model.pth')))
        model.train()  # Set the model in training mode

    diffusion = GaussianDiffusion(
        model,
        image_size=64,    # height and width of frames
        num_frames=10,    # number of video frames
        timesteps=1000,   # number of steps
        loss_type='l1'    # L1 or L2
    )

    # Assuming you have your optimizer defined
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of training iterations
    num_iterations = 100
    batch_size = 1
    checkpoint_interval = 80
    checkpoint_path = os.path.join(paths.model_dir, 'main_model.pth')
    dataset_size = 9  # Number of training videos

    for iteration in trange(num_iterations):

        # Sample indices for the current batch
        batch_indices = torch.randint(0, dataset_size, (batch_size,))
        
        # Sample a batch of training data
        batch_videos = train_tensor[batch_indices]
        batch_text = [train_text[idx] for idx in batch_indices]

        # Forward pass with text conditioning
        loss = diffusion(batch_videos, cond=batch_text)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration [{iteration}]: Loss = {loss.item()}")
        if iteration % 100 == 0:
            print(f"Train Iteration [{iteration}/{num_iterations}]: Loss = {loss.item()}")
            
        # Validation every checkpoint_interval iterations
        if (iteration + 1) % checkpoint_interval == 0:
            val_loss = 0.0
            num_val_batches = len(val_tensor) // batch_size

            for val_batch_start in range(0, len(val_tensor), batch_size):
                val_batch_videos = val_tensor[val_batch_start:val_batch_start + batch_size]
                val_batch_text = val_text[val_batch_start:val_batch_start + batch_size]

                with torch.no_grad():
                    val_batch_loss = diffusion(val_batch_videos, cond=val_batch_text)
                    val_loss += val_batch_loss.item()

            val_loss /= num_val_batches
            print(f"Validation Iteration [{iteration}/{num_iterations}]: Loss = {val_loss}")
            
            # Save checkpoint
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration + 1}")

    # Update the training Checkpoint

    print("Training finished!")

if __name__ == "__main__":
    main()
    