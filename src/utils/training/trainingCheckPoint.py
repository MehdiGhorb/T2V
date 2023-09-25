import yaml

def getLatestCheckpoint(csv_file_path):
    # Load the YAML content from the file into a Python dictionary
    with open(csv_file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Update the data in the Python dictionary
    return data["Total_iterations"]

