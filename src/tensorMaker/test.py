import torch
import yaml
import sys
import os

sys.path.append('../common')
import paths


# Read the config file
with open(os.path.join(paths.CONFIG_DIR, 'tensorConfig.yaml'), "r") as config:
    tensor_config = yaml.load(config, Loader=yaml.FullLoader)

# Create a sample tensor to compare and remove the unqualified tensors
size = (1000,
        int(tensor_config["channels"]), 
        int(tensor_config["frame_num"]), 
        int(tensor_config["frame_size"]/2), 
        int(tensor_config["frame_size"]/2))
sample_shape = torch.tensor(size)
tensor = sample_shape

print(torch.equal(torch.tensor([1000, 3, 10, 64, 64]), sample_shape))