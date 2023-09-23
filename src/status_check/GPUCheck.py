import subprocess

# Run the nvidia-smi command and capture its output
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True, shell=True)

# Check if the command was successful
if result.returncode == 0:
    gpu_info = result.stdout
    # Process the gpu_info as needed
else:
    print("Error running nvidia-smi")
