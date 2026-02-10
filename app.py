import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore

# --- LlamaIndex Imports ---
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, ExactMatchFilter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. API KEYS (Ideally use .env, but hardcoded for now)
# NOTE: Make sure these are correct!
GOOGLE_API_KEY = "AIzaSyCqaJbLgrMacnj416bJFSJZ3YlZ_PpMFE8"
PINECONE_API_KEY = "pcsk_65wY37_BMrtE8ZeUzz5xU1YaVw71sN1d66xJoXUU5qrkf6DRqD5GupumcFZUh5g4wyiVdt"
INDEX_NAME = "bmsit-chatbot"

# 2. APP SETUP
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. CONFIGURE AI (THE FIX IS HERE)
print("⚙️ Setting up Gemini & Embeddings...")

# --- CRITICAL FIX: Changed from 'text-embedding-004' to 'embedding-001' ---
embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)

# Try to get the best model, fallback to flash if pro fails
try:
    llm = GoogleGenAI(model="models/gemini-1.5-pro", api_key=GOOGLE_API_KEY)
except:
    print("⚠️ Pro model failed, switching to Flash...")
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)

Settings.llm = llm
Settings.embed_model = embed_model

# 4. CONNECT TO FIREBASE & PINECONE
# Check if firebase is already initialized to avoid "App already exists" error
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred)
        print("🔥 Firebase Connected!")
    except Exception as e:
        print(f"⚠️ Firebase Error (Check firebase_key.json): {e}")

db = firestore.client()

# Pinecone Connection
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 5. DATA MODELS
class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy (Default)"

# 6. MODES (PERSONAS)
PERSONAS = {
    "Study Buddy (Default)": (
        "You are a helpful, encouraging BMSIT student study partner. "
        "Keep answers clear, concise, and friendly."
    ),
    "The Professor 🎓": (
        "You are a strict, formal professor. Use advanced academic vocabulary. "
        "Do not tolerate simple mistakes. Be precise."
    ),
    "The Bro / Senior 🕶️": (
        "You are a chill senior student. Use slang (fam, lit, bet, dw). "
        "Be casual but helpful."
    ),
    "ELI5 (Simple) 👶": (
        "Explain like the user is 5 years old. Use simple analogies. "
        "Be very enthusiastic and keep it simple."
    )
}

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    """Simple check to see if server is running"""
    return {"status": "Online", "model": "Gemini + Embedding-001 (Stable)"}

@app.post("/chat")
def chat(request: ChatRequest):
    """The main chat function"""
    try:
        # Get the system prompt based on mode
        system_prompt = PERSONAS.get(request.mode, PERSONAS["Study Buddy (Default)"])
        
        # Filter logic
        # We ensure we only search the specific Year's documents
        filters = MetadataFilters(filters=[ExactMatchFilter(key="year", value=str(request.year))])
        
        # Create engine for this specific request
        # 'context' mode means it retrieves docs + chat history
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=system_prompt,
            filters=filters,
            similarity_top_k=5 
        )
        
        # Generate response
        response = chat_engine.chat(request.message)
        return {"response": str(response)}
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        # Return a friendly error message to the frontend instead of crashing
        return {"response": "My brain is having a hiccup! Please try again in a moment. 🤖"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
