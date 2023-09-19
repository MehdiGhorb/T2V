import torch    

def saveTensor(tensor_list, tensor_path):
    try:
        # Stack the tensors along a new dimension
        stacked_tensor = torch.stack(tensor_list, dim=0)
        
        # Save the stacked tensor to the file
        torch.save(stacked_tensor, tensor_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def loadTensor(tensor_path):
    # Load the tensor from the file
    return torch.load(tensor_path)
