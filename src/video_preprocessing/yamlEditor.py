import os
import yaml

def createOrLoadYamlFile(file_path):
    # Check if the YAML file exists
    if not os.path.exists(file_path):
        # Create a new YAML structure if the file doesn't exist
        data = {
            'Source': '',
            'failed_indexes': [],
            'video_checkpoint_from_to': [],
            'total_videos': 0
        }
        
        # Write the YAML data to the file
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
    
    # Load and return the YAML data from the file
    with open(file_path, 'r') as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
    
def loadMainYamlFile(file_path):
    # Check if the YAML file exists
    if not os.path.exists(file_path):
        # Create a new YAML structure if the file doesn't exist
        parts = file_path.split('/')
        data = {
            'Source': f'WebVid_{parts[-1].replace(".csv", "")}',
            'Total_video_checkpoint': 0,
            'Total_iterations': 0
        }
        
        # Write the YAML data to the file
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
    
    # Load and return the YAML data from the file
    with open(file_path, 'r') as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
    
def updateMainYamlFile(file_path, source: str, total_iterations: int, total_video_check_point: int):
    # Load the YAML content from the file into a Python dictionary
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Update the data in the Python dictionary
    data["Source"] = source
    data["Total_iterations"] = total_iterations
    data["Total_video_checkpoint"] = total_video_check_point

    # Write the updated data back to the YAML file
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def updateIterationYamlFile(file_path, source: str, failed_indexes: list[int], total_videos: int, video_checkpoint_from_to: list[int]):
    # Load the YAML content from the file into a Python dictionary
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Update the data in the Python dictionary
    data["Source"] = source
    data["failed_indexes"] = failed_indexes
    data["total_videos"] = total_videos
    data["video_checkpoint_from_to"] = video_checkpoint_from_to

    # Write the updated data back to the YAML file
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def extractKeysFromYaml(data, parent_key='', separator='_'):
    keys = []
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            keys.extend(extractKeysFromYaml(value, new_key, separator=separator))
        else:
            keys.append(new_key)
    return keys
