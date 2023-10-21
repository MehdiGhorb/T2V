from googleapiclient.discovery import build
from cloudUtils import *
import re

def mainModelBackupControl(folder_id, num_elements=5):
    # Authenticate
    creds = authenticate(os.path.join(paths.CLOUD_CREDS, 'token.json'))

    try:
        # Create Google Drive API client
        service = build('drive', 'v3', credentials=creds)

        # List files in the specified folder
        results = service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
        files = results.get('files', [])

        model_names = []
        for file in files:
            model_names.append(file['name'])
            #print(f"File Name: {file['name']}, File ID: {file['id']}")
        if len(model_names) > num_elements:
            min_number_file = min(model_names, 
                                  key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
            for file in files:
                if file['name'] == min_number_file:
                    delete_file(file_id=file['id'])
                    print(f"{file['name']} (backup) has been successfully deleted.\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def delete_file(file_id):
    # Authenticate
    creds = authenticate(os.path.join(paths.CLOUD_CREDS, 'token.json'))

    try:
        # Create Google Drive API client
        service = build('drive', 'v3', credentials=creds)

        # Delete the file by its ID
        service.files().delete(fileId=file_id).execute()
        print(f"File with ID {file_id} (backup) has been deleted.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
folder_id = '1D9FY29prg_G2GdGrl4joI6HK0yWL2ZRY'
mainModelBackupControl(folder_id)

