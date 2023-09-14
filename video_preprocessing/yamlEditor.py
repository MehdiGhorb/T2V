import os
import yaml
import re

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
    
def extractKeysFromYaml(data, parent_key='', separator='_'):
    keys = []
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            keys.extend(extractKeysFromYaml(value, new_key, separator=separator))
        else:
            keys.append(new_key)
    return keys
