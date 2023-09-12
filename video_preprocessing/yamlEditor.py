import os
import yaml
import re

def createOrLoadYamlFile(file_path):
    # Check if the YAML file exists
    if not os.path.exists(file_path):
        # Create a new YAML structure if the file doesn't exist
        data = {
            'Source': '',
            'failed_indexes_0': [],
            'total_video_checkpoint': 0,
            'total_videos_0': 0
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

def findLargestIndexedString(strings, format_pattern=r'failed_indexes_(\d+)'):
    # Initialize variables to store the largest index and the corresponding string
    largest_index = -1
    largest_string = None

    for string in strings:
        # Use regular expression to extract the index
        match = re.search(format_pattern, string)
        if match:
            index = int(match.group(1))  # Extract and convert the index to an integer

            # Check if the current index is larger than the previous largest
            if index > largest_index:
                largest_index = index
                largest_string = string

    return largest_string
