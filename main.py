import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import os
from tqdm import trange
sys.path.append('/helper')
from video_annotation import read_data
sys.path.append('/tensorMaker')
from saveTensor import loadTensor

rows = read_data(csv_file_path)

video_description = []
# Read video descriptions
for index in rows:
  video_description.append(index[4])

#torch.cuda.empty_cache()
model = Unet3D(
    dim=64,
    use_bert_text_cond=True,  # this must be set to True to auto-use the bert model dimensions
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,    # height and width of frames
    num_frames=10,     # number of video frames
    timesteps=1000,   # number of steps
    loss_type='l1'    # L1 or L2
)

''' Adding validation dataset '''
val_videos = torch.randn(30, 3, 10, 32, 32)
val_text = ["text"] * 30

# Assuming you have your optimizer defined
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of training iterations
num_iterations = 100
batch_size = 16
checkpoint_interval = 80
checkpoint_path = "model_checkpoint.pth"
dataset_size = 100  # Number of training videos

for iteration in trange(num_iterations):

    # Sample indices for the current batch
    batch_indices = torch.randint(0, dataset_size, (batch_size,))

    # Load Tensor
    stacked_tensor = loadTensor()
    # Sample a batch of training data
    batch_videos = stacked_tensor[batch_indices]
    batch_text = [video_description[idx] for idx in batch_indices]

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
        num_val_batches = len(val_videos) // batch_size

        for val_batch_start in range(0, len(val_videos), batch_size):
            val_batch_videos = val_videos[val_batch_start:val_batch_start + batch_size]
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

print("Training finished!")
