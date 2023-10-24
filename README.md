# T2V

## Welcome to automated training software with diffusion models on WebVid dataset 

### This Software contains two main parts:

  1- Video Preprocessing (Including downloads, processing, uploading, and ...)
  
  2- Training

### Important scripts

1- videoTensor.py

2- main.py

3- If you wish to use LABS or Jupyter Notebook, use main_training.ipynb

### How to use the Software

To Use the Software you first need to prepare video Tensors of shape (1000, 20, 3, 256, 256)

1000: Number of videos in the Tensor (This number can be flexible)

20: Number of frames

3: Number of channels

256: width x size

To do so simply run the following command in 'src/exe_operations'

```
python3 videoTensor.py customised_results_2M_train.csv [The number of videos you want each Tensor to contain] [The number of Tensors you want to create]

python3 videoTensor.py customised_results_2M_train.csv 1000 500
```

The above code will create 500 Tensors each containing 1000 videos



  ```python
def hello_world():
    print("Hello, world!")
```
