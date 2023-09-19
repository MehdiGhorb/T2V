import io
import torch    

def saveBuffer(tensor_list, tensor_path):
    # Stack the tensors along a new dimension
    stacked_tensor = torch.stack(tensor_list, dim=0)
    # Save to file
    torch.save(torch.stack(tensor_list, dim=0), tensor_path)
    # Save to io.BytesIO buffer
    buffer = io.BytesIO()
    torch.save(stacked_tensor, buffer)
    # Close the buffer to release its resources
    buffer.close()

def loadBuffer(tensor_path):
    # Load the tensor from the file
    return torch.load(tensor_path)
