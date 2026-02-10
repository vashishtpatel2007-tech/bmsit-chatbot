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

# 2. CONFIGURE AI (EXPLICIT PURGE)
print("🚀 [STATUS] FORCING EMBEDDING-001 AND RESTORING MODES")

try:
    # Stable model to bypass billing/version issues
    embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    
    # Global Defaults
    Settings.embed_model = embed_model
    Settings.llm = llm
except Exception as e:
    print(f"❌ AI CONFIG ERROR: {e}")

# 3. INITIALIZE APP
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATA ---
DATABASE = {
    "1": "https://drive.google.com/drive/folders/1Yv-tfstUnQytvhvdLP02j6IDiolovIWI?usp=drive_link",
    "2": "https://drive.google.com/drive/folders/1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp?usp=drive_link",
    "3": "https://drive.google.com/drive/folders/1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e?usp=drive_link",
    "4": "https://drive.google.com/drive/folders/17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1?usp=drive_link"
}

# 🎭 STUDY MODES (RESTORED)
PERSONA_RULES = {
    "Study Buddy": "You are 'Alex', an energetic BMSIT senior. VIBE: Positive, supportive, uses emojis. 🚀",
    "The Professor": "You are Professor Sharma. VIBE: Formal, academic, precise. No slang.",
    "The Bro": "You are 'Sam', the campus legend. VIBE: Casual, chill, uses slang (fam, bet, dw). 🕶️",
    "ELI5": "You are a patient tutor. VIBE: Explains hard things simply using analogies for a 5-year-old."
}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        print(f"📩 Mode: {request.mode} | Year: {request.year} | Msg: {request.message}")
        
        year = str(request.year) if str(request.year) in DATABASE else "1"
        drive_link = DATABASE[year]
        persona = PERSONA_RULES.get(request.mode, PERSONA_RULES["Study Buddy"])
        
        # System Prompt construction
        system_prompt = (
            f"{persona}\n\n"
            f"You are helping a Year {year} student at BMSIT.\n"
            f"RESOURCES DRIVE: {drive_link}\n"
            "If they ask for files/notes, give them the link. Otherwise, answer from context."
        )

        # 4. CONNECT TO BRAIN
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Pass embed_model directly to index and engine to block 404 errors
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=year)]),
            system_prompt=system_prompt,
            embed_model=embed_model 
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}

    except Exception as e:
        print(f"❌ CHAT CRASH: {e}")
        return {"response": "My brain is having a hiccup! Please try again. 🤖"}

@app.get("/")
def home():
    return {"status": "Online", "message": "ALL MODES ACTIVE ✅"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
