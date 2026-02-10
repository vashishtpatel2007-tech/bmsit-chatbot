import os
import time
import shutil
import io
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
INDEX_NAME = "bmsit-chatbot"

# YOUR FOLDER MAP
folder_map = {
    "1": "1Yv-tfstUnQytvhvdLP02j6IDiolovIWI", 
    "2": "1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp",
    "3": "1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e", 
    "4": "17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1"
}

# --- 2. UNIVERSAL INSTRUCTION ---
UNIVERSAL_INSTRUCTION = """
You are a universal document digitizer.
1. TIMETABLES: Convert grid data into a list: "Day - Time - Subject".
2. METADATA: If you find a link to the original file, preserve it.
3. HANDWRITING: Transcribe exactly.
"""

# --- 3. SETUP AI ---
if not GOOGLE_API_KEY or not PINECONE_API_KEY or not LLAMA_CLOUD_API_KEY:
    print("❌ KEYS MISSING. Check .env file.")
    exit()

print("⚙️ configuring AI (Universal Vision Mode)...")

try:
    # FIXED: Switched to 'embedding-001' to fix 404 error
    embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Configure Parser
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="text",
        system_prompt=UNIVERSAL_INSTRUCTION, 
        premium_mode=True,
        verbose=True
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
    print("\n🚀 STARTING UPDATE (With File Links)...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        # Assumes credentials.json is created by GitHub Actions
        creds = Credentials.from_service_account_file("credentials.json")
        service = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Drive Connection Failed: {e}")
        return

    if not os.path.exists("temp_downloads"):
        os.makedirs("temp_downloads")

    total_docs = 0

    for year, folder_id in folder_map.items():
        print(f"\n📂 Processing Year {year}...")
        
        try:
            results = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="files(id, name, mimeType, webViewLink)"
            ).execute()
            items = results.get('files', [])

            if not items:
                print("   ⚠️ No files found.")
                continue

            documents_to_upload = []
            
            for item in items:
                file_id = item['id']
                file_name = item['name']
                web_link = item.get('webViewLink', '')
                
                if "application/pdf" in item['mimeType']:
                    print(f"   ⬇️  Linking & Parsing: {file_name}...")
                    local_path = os.path.join("temp_downloads", file_name)
                    
                    try:
                        download_file(service, file_id, local_path)
                        parsed_results = parser.load_data(local_path)
                        
                        for doc in parsed_results:
                            # Instead of editing the locked 'doc.text', we create a NEW Document
                            combined_text = doc.text + f"\n\n[OFFICIAL DOCUMENT LINK]: {web_link}"
                            
                            new_metadata = {
                                "year": year, 
                                "file_name": file_name,
                                "file_link": web_link
                            }
                            
                            # Create a fresh, unlocked document
                            new_doc = Document(text=combined_text, metadata=new_metadata)
                            documents_to_upload.append(new_doc)

                        os.remove(local_path)
                        
                    except Exception as e:
                        print(f"   ❌ Failed to parse {file_name}: {e}")

            if documents_to_upload:
                print(f"   📄 Uploading {len(documents_to_upload)} pages with links...")
                VectorStoreIndex.from_documents(documents_to_upload, storage_context=storage_context)
                print(f"   ✅ Year {year} Updated!")
                total_docs += len(documents_to_upload)
        
        except Exception as e:
            if "429" in str(e):
                print("   ⏳ Speed Limit Hit. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"   ❌ Error Year {year}: {e}")

    if os.path.exists("temp_downloads"):
        shutil.rmtree("temp_downloads")
    print(f"\n🎉 SUCCESS! Added {total_docs} pages. Links are active.")

if __name__ == "__main__":
    update_database()
