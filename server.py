import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore

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

# 2. CONFIGURE AI (THE FIX)
# We are printing this so we can prove it's running in the Render logs
print("🚀 [CRITICAL UPDATE] SERVER STARTING WITH MODELS/EMBEDDING-001")

try:
    # Use embedding-001 strictly to avoid the 404 error
    embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = llm
except Exception as e:
    print(f"❌ AI CONFIG ERROR: {e}")

# 3. INITIALIZE FIREBASE (Fixing the previous crash)
if not firebase_admin._apps:
    try:
        # Assumes firebase_key.json is in your root folder
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred)
        print("🔥 Firebase Initialized Successfully")
    except Exception as e:
        print(f"⚠️ Firebase Init Warning (Optional): {e}")

# 4. INITIALIZE APP
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE = {
    "1": "https://drive.google.com/drive/folders/1Yv-tfstUnQytvhvdLP02j6IDiolovIWI?usp=drive_link",
    "2": "https://drive.google.com/drive/folders/1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp?usp=drive_link",
    "3": "https://drive.google.com/drive/folders/1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e?usp=drive_link",
    "4": "https://drive.google.com/drive/folders/17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1?usp=drive_link"
}

PERSONA_RULES = {
    "Study Buddy": "You are 'Alex', a supportive senior. VIBE: Positive, emojis. GOAL: Help the student.",
    "The Professor": "You are Professor Sharma. VIBE: Professional, precise. GOAL: Provide accurate info.",
    "The Bro": "You are 'Sam', the chillest guy. VIBE: Slang, casual. GOAL: Quick answers.",
    "ELI5": "Explain like I'm 5. VIBE: Simple analogies."
}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"
    token: str = None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        year = str(request.year) if str(request.year) in DATABASE else "1"
        drive_link = DATABASE[year]
        persona = PERSONA_RULES.get(request.mode, PERSONA_RULES["Study Buddy"])
        
        system_prompt = (
            f"{persona}\n\n"
            f"OFFICIAL DRIVE LINK: {drive_link}\n"
            "PROTOCOL: If asked for files, share the Drive Link. If asked for info, answer from context."
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=year)]),
            system_prompt=system_prompt
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}

    except Exception as e:
        print(f"❌ CHAT CRASH LOG: {e}")
        return {"response": "My brain is having a hiccup! Try again in a sec. 🤖"}

@app.get("/")
def home():
    return {"status": "Active", "message": "SERVER.PY IS RUNNING ✅"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
