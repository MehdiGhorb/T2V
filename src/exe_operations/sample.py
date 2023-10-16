import sys
import yaml
import os
import torch
from video_diffusion_pytorch import GaussianDiffusion, Unet3D

sys.path.append('../common')
import paths

'''Load training parameters'''
with open(os.path.join(paths.CONFIG_DIR, 'trainingParams.yaml'), 'r') as f:
    training_params = yaml.safe_load(f)

'''Load Tensor parameters'''
with open(os.path.join(paths.CONFIG_DIR, 'tensorConfig.yaml'), 'r') as f:
    tensor_params = yaml.safe_load(f)

'''Tensor Parameters'''
IMAGE_SIZE = int(tensor_params['frame_size']/2)
NUM_FRAMES = int(tensor_params['frame_num'])

'''Training parameters'''
NUM_ITERATIONS = int(training_params['num_iterations'])
TIME_STEPS = int(training_params['time_steps'])
TRAINING_LOSS_TYPE = training_params['training_loss_type']

'''Model'''
model = Unet3D(
    dim=64,
    use_bert_text_cond=True,  # this must be set to True to auto-use the bert model dimensions
    dim_mults=(1, 2, 4, 8),
)

# Step 2: Load Model Weights
checkpoint_path = paths.MAIN_MODEL  # Provide the path to your saved model checkpoint
checkpoint = torch.load(checkpoint_path)

# If the checkpoint contains the entire model
model.load_state_dict(checkpoint['model_state_dict'])

# If the checkpoint contains the optimizer state, use this to load the optimizer's state_dict:
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Step 3: Set the Model in Evaluation Mode (if using it for inference)
model.eval()

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,          # height and width of frames
    num_frames=NUM_FRAMES,          # number of video frames
    timesteps=TIME_STEPS,           # number of steps
    loss_type=TRAINING_LOSS_TYPE    # L1 or L2
)

new_text = [
    'a woman face moving and looking around'
]

sampled_videos = diffusion.sample(cond=new_text, cond_scale=2)
print(sampled_videos.shape)
