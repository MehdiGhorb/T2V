import sys
import yaml
import os
import argparse
import torch
from video_diffusion_pytorch import GaussianDiffusion, Unet3D
from torchvision import transforms as T

sys.path.append('../common')
import paths

'''TEMP - To Be Deleted'''
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, 
                   save_all = True, 
                   append_images = rest_imgs, 
                   duration = duration, 
                   loop = loop, 
                   optimize = optimize)
    return images

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

# Load Model Weights
checkpoint_path = paths.MAIN_MODEL  # Provide the path to your saved model checkpoint
checkpoint = torch.load(checkpoint_path)

# If the checkpoint contains the entire model
model.load_state_dict(checkpoint['model_state_dict'])

# If the checkpoint contains the optimizer state, use this to load the optimizer's state_dict:
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Set the Model in Evaluation Mode (if using it for inference)
model.eval()

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,          # height and width of frames
    num_frames=NUM_FRAMES,          # number of video frames
    timesteps=TIME_STEPS,           # number of steps
    loss_type=TRAINING_LOSS_TYPE    # L1 or L2
)

def main():
    parser = argparse.ArgumentParser(description='Main Training Script')
    parser.add_argument('prompt', help='Path to the CSV file containing video descriptions')
    args = parser.parse_args()
    # Sample
    sampled_video = diffusion.sample(cond=[args.prompt], cond_scale=2)

    #Convert the Tensor to a GIF
    video_tensor_to_gif(sampled_video[0], f"{paths.SAMPLES_DIR}/{args.prompt}.gif")

    print(f"The sample was created with a size of {sampled_video.shape} and was saved at {paths.SAMPLES_DIR}.")

if __name__ == "__main__":
    main()
    