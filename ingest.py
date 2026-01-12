import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.google import GoogleDriveReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from pinecone import Pinecone

# --- IMPORTS ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. Load keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    print("❌ Error: Keys not found! Check your .env file.")
    exit()

# 2. Configure Brain (Gemini 2.0 Flash)
try:
    print("⚙️ Configuring AI Models...")
    # Using the model we found in your account
    Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    print("   ✅ Using Model: Gemini 2.0 Flash")
except Exception as e:
    print(f"❌ Critical Error configuring AI: {e}")
    exit()

# 3. Folder Map (Updated with your links)
folder_map = {
    "1": "1Yv-tfstUnQytvhvdLP02j6IDiolovIWI", 
    "2": "1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp",
    "3": "1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e",
    "4": "17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1"
}

def load_and_index():
    print("🚀 Starting Data Ingestion...")

    # Connect to Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "bmsit-chatbot"
        
        # Create Index if needed
        existing_indexes = [i.name for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"⚙️ Creating Pinecone Index: {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=768, 
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
        
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return
    
    all_documents = []

    # Connect to Drive (THE FIX IS HERE)
    try:
        # We specifically tell it this is a Service Account Key
        loader = GoogleDriveReader(service_account_key_path="credentials.json")
    except Exception as e:
        print(f"❌ Error reading credentials.json: {e}")
        return

    # Loop Folders
    for year, folder_id in folder_map.items():
        print(f"📂 Scanning Year {year} folder...")
        try:
            docs = loader.load_data(folder_id=folder_id)
            for doc in docs:
                doc.metadata = {"year": year}
                all_documents.extend([doc])
            print(f"   ✅ Found {len(docs)} pages.")
        except Exception as e:
            print(f"   ❌ ERROR for Year {year}: {e}")
            print("      (Make sure you shared the folder with the Service Account email!)")

    if not all_documents:
        print("❌ No documents found. Stopping.")
        return

    # Upload
    print("🧠 Uploading to Pinecone...")
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(all_documents, storage_context=storage_context)
        print("🎉 SUCCESS! The Data is live in the database.")
    except Exception as e:
        print(f"❌ Upload Error: {e}")

if __name__ == "__main__":
    load_and_index()