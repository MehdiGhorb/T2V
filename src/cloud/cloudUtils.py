from __future__ import print_function
import sys
import os.path
from tqdm import tqdm
import yaml

sys.path.append('../common')
import paths

sys.path.append(paths.DOWNLOADER_PATH)
from yamlEditor import updateIterationYamlFile

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.credentials import Credentials

DATA_TYPE = {
    'tensor' : 'application/octet-stream',
    'yaml' : 'application/x-yaml',
}

def authenticate(token):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, SCOPES)
        return creds
    else:
        raise FileNotFoundError

def uploadYAML(file_path, folder_id, file_name_to_upload, tensor_id=None):
    """Insert a new file into a specific folder on Google Drive with an optional new name.
    Returns: Id of the file uploaded.
    """
    # Authenticate
    creds = authenticate(os.path.join(paths.CLOUD_CREDS, 'token.json'))

    # Add Tensor ID to track yaml
    if tensor_id != None:
        with open(file_path) as temp_yaml:
            track_yaml = yaml.load(temp_yaml, Loader=yaml.FullLoader)
        track_yaml['track_tensor_cloud_ID'] = tensor_id  # Add the temporary file ID to the YAML data
        with open(file_path, 'w') as updated_yaml_file:
            yaml.dump(track_yaml, updated_yaml_file, default_flow_style=False)

    try:
        # Create drive API client
        service = build('drive', 'v3', credentials=creds)
        # Create the file with a temporary name
        file_metadata = {
            'name': file_name_to_upload,
            'parents': [folder_id]  # Specify the folder ID here
        }
        media = MediaFileUpload(file_path, mimetype=DATA_TYPE['yaml'])
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()

        print(F'YAML File (ID: {file.get("id")}) upload Complete.')

    except HttpError as e:
        print(F'An error occurred: {str(e)}')

def uploadTensor(file_path, folder_id, file_name_to_upload):
    # Authenticate
    creds = authenticate(os.path.join(paths.CLOUD_CREDS, 'token.json'))

    try:
        # Create drive API client
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {
            'name': file_name_to_upload,
            'parents': [folder_id]  # Specify the folder ID here
        }

        media = MediaFileUpload(file_path, mimetype=DATA_TYPE['tensor'])
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        file_id = file.get("id")  # Get the Tensor cloud ID

        print(F'Tensor File (ID: {file_id}) upload Complete.')

    except HttpError as e:
        print(F'An error occurred: {str(e)}')
    
    return file_id      

def downloadFile(file_id, save_path):
    """Downloads a file
    Args:
        real_file_id: ID of the file to download
        save_path: Path to save the downloaded file
    Returns: True if download is successful, False otherwise
    """

    token = os.path.join(paths.CLOUD_CREDS, 'token.json')
    creds = authenticate(token)

    try:
        # Create drive API client
        service = build('drive', 'v3', credentials=creds)

        # Get the file's metadata to determine its name
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name', 'untitled')

        # Create a stream to save the downloaded file
        with open(os.path.join(save_path, file_name), 'wb') as local_file:
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(local_file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(F'Download {int(status.progress() * 100)}.')

        return True

    except HttpError as error:
        print(F'An error occurred: {error}')
        return False

def uploadDir(root_folder_path, parent_folder_id=None):
    """Uploads a directory with its subdirectories and files, preserving the structure.
    Returns: ID of the parent folder on Google Drive
    """
    token = os.path.join(paths.CLOUD_CREDS, 'token.json')
    creds = authenticate(token)
    parent_folder_id = parent_folder_id

    try:
        # Create drive API client
        service = build('drive', 'v3', credentials=creds)

        if parent_folder_id is None:
            # If no parent folder ID is provided, create a new folder on Google Drive
            root_folder_name = os.path.basename(root_folder_path)
            # Check if a folder with the same name already exists
            query = f"name='{root_folder_name}' and mimeType='application/vnd.google-apps.folder'"
            results = service.files().list(q=query).execute()
            existing_folders = results.get('files', [])

            if existing_folders:
                # Use the first existing folder if found
                parent_folder_id = existing_folders[0]['id']
            else:
                # Create a new folder if no existing folder is found
                folder_metadata = {
                    'name': root_folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = service.files().create(body=folder_metadata, fields='id').execute()
                parent_folder_id = folder.get('id')

        for root, dirs, files in os.walk(root_folder_path):
            # Create subdirectories in Google Drive
            for dir_name in dirs:
                dir_metadata = {'name': dir_name, 'parents': [parent_folder_id], 'mimeType': 'application/vnd.google-apps.folder'}
                dir_file = service.files().create(body=dir_metadata, fields='id').execute()
                dir_id = dir_file.get('id')

                # Upload files in subdirectory
                for filename in tqdm(os.listdir(os.path.join(root, dir_name)), desc=dir_name):
                    file_path = os.path.join(root, dir_name, filename)
                    file_metadata = {'name': filename, 'parents': [dir_id]}
                    media = MediaFileUpload(file_path)
                    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    except HttpError as error:
        print(F'An error occurred: {error}')

    return parent_folder_id

def updateFile(file_name, new_content_path):
    token = os.path.join(paths.CLOUD_CREDS, 'token.json')
    creds = authenticate(token)
    try:
        # Create Google Drive API client
        service = build('drive', 'v3', credentials=creds)

        # Find the existing file by name
        results = service.files().list(q=f"name='{file_name}'").execute()
        files = results.get('files', [])

        if not files:
            print(f"No file with the name '{file_name}' found.")
            return

        # Assume we are updating the first matching file, but you might need to specify further criteria
        file = files[0]

        # Update the file's content
        media = MediaFileUpload(new_content_path, mimetype=file['mimeType'])
        service.files().update(
            fileId=file['id'],
            media_body=media
        ).execute()

        print(f"File '{file_name}' updated successfully.")

    except HttpError as error:
        print(f'An error occurred: {error}')

# Function to get folder ID by name
def getFolderIDByName(folderYAML, name):
    for folder in folderYAML['folders']:
        if folder['name'] == name:
            return folder['id']
    return None
