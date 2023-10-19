import os
import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload

sys.path.append('../common')
import paths

def authenticate(token):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, SCOPES)
        return creds
    else:
        raise FileNotFoundError

def downloadFile(file_name, file_id, save_path):
    """Downloads a file
    Args:
        real_file_id: ID of the file to download
        save_path: Path to save the downloaded file
    Returns: True if download is successful, False otherwise
    """
    print(f'Downloading {file_name} ...\n')

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
            print(f'{file_name} was downloaded successfully\n')

        return True

    except HttpError as error:
        print(F'An error occurred: {error}')
        return False
    
downloadFile('1PzsTvhJ-hm0mMafWfc-ZqLhr2MHNQnMm', '')