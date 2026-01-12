from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. HARDCODED KEYS
GOOGLE_API_KEY = "AIzaSyCqaJbLgrMacnj416bJFSJZ3YlZ_PpMFE8"
PINECONE_API_KEY = "pcsk_65wY37_BMrtE8ZeUzz5xU1YaVw71sN1d66xJoXUU5qrkf6DRqD5GupumcFZUh5g4wyiVdt"

# 2. APP SETUP
app = FastAPI()

# Allow the frontend to talk to this backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. CONFIGURE AI
embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
try:
    llm = GoogleGenAI(model="models/gemini-1.5-pro-latest", api_key=GOOGLE_API_KEY)
except:
    llm = GoogleGenAI(model="models/gemini-flash-latest", api_key=GOOGLE_API_KEY)

Settings.llm = llm
Settings.embed_model = embed_model

# 4. CONNECT TO FIREBASE & PINECONE
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("bmsit-chatbot")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# Load index once at startup
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# 5. DATA MODELS (The "Rules" for talking to the API)
class ChatRequest(BaseModel):
    message: str
    year: str
    mode: str

# 6. MODES
PERSONAS = {
    "Study Buddy (Default)": "You are a helpful, encouraging BMSIT student study partner. Keep answers clear, concise, and friendly.",
    "The Professor 🎓": "You are a strict, formal professor. Use advanced academic vocabulary. Do not tolerate simple mistakes.",
    "The Bro / Senior 🕶️": "You are a chill senior student. Use slang (fam, lit, bet). Be casual.",
    "ELI5 (Simple) 👶": "Explain like the user is 5 years old. Use simple analogies. Be very enthusiastic."
}

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    """Simple check to see if server is running"""
    return {"status": "Online", "model": "Gemini 1.5"}

@app.post("/chat")
def chat(request: ChatRequest):
    """The main chat function"""
    try:
        # Get the system prompt based on mode
        system_prompt = PERSONAS.get(request.mode, PERSONAS["Study Buddy (Default)"])
        
        # Filter logic (Same as before)
        filters = MetadataFilters(filters=[MetadataFilter(key="year", value=request.year)])
        
        # Create engine for this specific request
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=system_prompt,
            filters=filters
        )
        
        # Generate response
        response = chat_engine.chat(request.message)
        return {"response": str(response)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)