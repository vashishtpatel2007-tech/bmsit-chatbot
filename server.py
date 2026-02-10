import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- LlamaIndex Imports ---
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# 1. LOAD KEYS
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bmsit-chatbot")

# 2. CONFIGURE AI (EXPLICIT OVERRIDE)
print("🚀 [CRITICAL UPDATE] FORCING MODELS/EMBEDDING-001 INTO ALL ENGINES")

try:
    # We define the stable model strictly here
    embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    
    # Force into Global Settings
    Settings.embed_model = embed_model
    Settings.llm = llm
except Exception as e:
    print(f"❌ AI CONFIG ERROR: {e}")

# 3. INITIALIZE APP
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE = {
    "1": "https://drive.google.com/drive/folders/1Yv-tfstUnQytvhvdLP02j6IDiolovIWI?usp=drive_link",
    "2": "https://drive.google.com/drive/folders/1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp?usp=drive_link",
    "3": "https://drive.google.com/drive/folders/1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e?usp=drive_link",
    "4": "https://drive.google.com/drive/folders/17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1?usp=drive_link"
}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"
    token: str = None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # LOGGING TO PROVE THE CODE IS RUNNING
        print(f"📩 Incoming Request: {request.message} for Year {request.year}")
        
        year = str(request.year) if str(request.year) in DATABASE else "1"
        
        # CONNECT TO PINECONE
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # --- THE FIX: Pass embed_model EXPLICITLY to the Index ---
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model # <--- This kills the ghost
        )

        # --- THE FIX: Pass embed_model EXPLICITLY to the Query Engine ---
        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=year)]),
            embed_model=embed_model # <--- This ensures the query doesn't use 004
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}

    except Exception as e:
        print(f"❌ CHAT CRASH LOG: {e}")
        return {"response": "My brain is having a hiccup! Please try again in a sec. 🤖"}

@app.get("/")
def home():
    return {"status": "Active", "message": "SERVER LIVE WITH FIXED EMBEDDINGS ✅"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
