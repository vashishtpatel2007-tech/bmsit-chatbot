from google.oauth2 import service_account
from googleapiclient.discovery import build

# 1. Setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = '1Yv-tfstUnQytvhvdLP02j6IDiolovIWI' # Your Year 1 ID

try:
    # 2. Connect
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # 3. List Files
    print(f"👀 Robot is looking inside folder: {FOLDER_ID}...")
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents",
        fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    if not items:
        print("❌ Result: The robot sees EMPTY folder (0 files).")
    else:
        print("✅ Result: The robot sees these files:")
        for item in items:
            print(f"   - {item['name']} ({item['mimeType']})")

except Exception as e:
    print(f"❌ Error: {e}")