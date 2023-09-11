from video_diffusion_pytorch import Unet3D, GaussianDiffusion
sys.path.append('/helper')
from vide_annotation import *

new_text = [
    'a woman face moving and looking around'
]

sampled_videos = diffusion.sample(cond=new_text, cond_scale=2)
print(sampled_videos.shape)