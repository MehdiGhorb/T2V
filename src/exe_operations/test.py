from video_diffusion_pytorch import Unet3D
import torch

# Model
input_shape = (1000, 3, 20, 128, 128)
model = Unet3D(
    dim=64,
    use_bert_text_cond=True,
    dim_mults=(1, 2, 4, 8),
)

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Calculate memory usage by considering input shape
input_size = torch.prod(torch.tensor(input_shape)).item()  # Calculate the total number of elements in the input
memory_usage_gb = ((total_params + input_size) * 4) / (1024**3)  # Assuming 4 bytes per parameter and input element

print(f"Total parameters in the model: {total_params}")
print(f"Memory usage with input shape: {memory_usage_gb:.2f} GB")
