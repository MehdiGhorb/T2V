from __future__ import print_function
import sys
import os.path
from tqdm import tqdm

sys.path.append('../common')
import paths
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

def uploadFile(file_name, data_type):
    """Insert new file.
    Returns : Id's of the file uploaded
    """
    token = os.path.join(paths.CLOUD_CREDS, 'token.json')
    creds = authenticate(token)
    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {'name': file_name}
        media = MediaFileUpload(file_name,
                                mimetype=data_type)
        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata, media_body=media,
                                      fields='id').execute()
        print(F'File ID: {file.get("id")}')

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    return file.get('id')


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

import os

import os

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


if __name__ == '__main__':
    uploadDir(root_folder_path='../../data/training_checkpoint')