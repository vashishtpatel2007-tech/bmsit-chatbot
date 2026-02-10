import os
import time
import shutil
import io
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_parse import LlamaParse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# --- 1. CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bmsit-chatbot")

# YOUR FOLDER MAP
folder_map = {
    "1": "1Yv-tfstUnQytvhvdLP02j6IDiolovIWI", 
    "2": "1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp",
    "3": "1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e", 
    "4": "17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1"
}

# --- 2. SETUP AI ---
if not GOOGLE_API_KEY:
    print("❌ KEYS MISSING. Check .env file.")
    exit()

print("⚙️ Configuring Gemini 2.0 Flash...")

try:
    # MATCHING YOUR SERVER CONFIG
    embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # LlamaParse
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="text",
        premium_mode=True
    )
except Exception as e:
    print(f"❌ AI Setup Failed: {e}")
    exit()

def download_file(service, file_id, filename):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(filename, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return filename

def update_database():
    print("\n🚀 STARTING UPDATE (Gemini 2.0)...")
    
    # --- CRITICAL: PRINT EMAIL FOR SHARING ---
    try:
        with open("credentials.json") as f:
            creds_data = json.load(f)
            client_email = creds_data.get('client_email')
            print(f"\n⚠️  IMPORTANT: Share Drive Folders with this email:")
            print(f"👉  {client_email}  👈")
            print("   (Give 'Viewer' access or I will find 0 files!)\n")
            time.sleep(5) 
    except:
        print("⚠️  Could not read credentials.json.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        creds = Credentials.from_service_account_file("credentials.json")
        service = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Drive Connection Failed: {e}")
        return

    if not os.path.exists("temp_downloads"):
        os.makedirs("temp_downloads")

    total_docs = 0

    for year, folder_id in folder_map.items():
        print(f"📂 Scanning Year {year}...")
        
        try:
            results = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="files(id, name, mimeType, webViewLink)"
            ).execute()
            items = results.get('files', [])

            if not items:
                print("   ⚠️  0 FILES FOUND. Share folder with email above.")
                continue
            
            documents_to_upload = []
            
            for item in items:
                if "application/pdf" in item['mimeType']:
                    print(f"   ⬇️  Found: {item['name']}")
                    local_path = os.path.join("temp_downloads", item['name'])
                    download_file(service, item['id'], local_path)
                    
                    try:
                        parsed_docs = parser.load_data(local_path)
                        for doc in parsed_docs:
                            doc.text += f"\n\n[LINK]: {item['webViewLink']}"
                            doc.metadata = {"year": year, "file_link": item['webViewLink']}
                            documents_to_upload.append(doc)
                        os.remove(local_path)
                    except Exception as e:
                        print(f"   ❌ Parse Error: {e}")

            if documents_to_upload:
                print(f"   📄 Uploading {len(documents_to_upload)} pages...")
                VectorStoreIndex.from_documents(documents_to_upload, storage_context=storage_context)
                print(f"   ✅ Year {year} Done!")
                total_docs += len(documents_to_upload)

        except Exception as e:
            print(f"   ❌ Error: {e}")

    if os.path.exists("temp_downloads"):
        shutil.rmtree("temp_downloads")
    print(f"\n🎉 SUCCESS! Added {total_docs} pages.")

if __name__ == "__main__":
    update_database()
