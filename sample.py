from video_diffusion_pytorch import GaussianDiffusion

# Load Tensor
model = loadModel()

diffusion = GaussianDiffusion(model)

new_text = [
    'a woman face moving and looking around'
]

sampled_videos = diffusion.sample(cond=new_text, cond_scale=2)
print(sampled_videos.shape)
