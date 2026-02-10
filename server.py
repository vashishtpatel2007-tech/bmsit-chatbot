import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- LlamaIndex Imports (OPENAI VERSION) ---
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# 1. LOAD KEYS
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bmsit-chatbot")

# 2. CONFIGURE AI (STABLE OPENAI)
print("🚀 MIGRATION COMPLETE: RUNNING GPT-4o-MINI")

try:
    # Industry standard stable models
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
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

# 🎭 ALL MODES RESTORED IN DETAIL
PERSONAS = {
    "Study Buddy": (
        "You are 'Alex', an energetic and super-supportive BMSIT senior. "
        "VIBE: Positive, encouraging, high energy! Use emojis like 🚀, ✨, 📚. "
        "GOAL: Make the student feel confident and capable."
    ),
    "The Professor": (
        "You are Professor Sharma, a distinguished academic at BMSIT. "
        "VIBE: Formal, helpful, professional, and very precise. "
        "GOAL: Provide high-quality, technically accurate information. No slang."
    ),
    "The Bro": (
        "You are 'Sam', the chillest guy on the BMSIT campus. "
        "VIBE: Casual, uses slang (fam, bet, easy scene, dw). "
        "GOAL: Give the answer straight with a relaxed vibe. 🕶️"
    ),
    "ELI5": (
        "You are a patient BMSIT Tutor. "
        "VIBE: Very gentle, slow, and extremely clear. "
        "GOAL: Explain complex engineering concepts using simple analogies for a 5-year-old."
    )
}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        print(f"📩 OpenAI Request: {request.message} (Mode: {request.mode}, Year: {request.year})")
        
        year = str(request.year) if str(request.year) in DATABASE else "1"
        drive_link = DATABASE[year]
        persona_rule = PERSONAS.get(request.mode, PERSONAS["Study Buddy"])
        
        system_prompt = (
            f"{persona_rule}\n\n"
            f"You are assisting a Year {year} student.\n"
            f"BMSIT DRIVE RESOURCES: {drive_link}\n\n"
            "If the user asks for notes, syllabus, or files, give them the Drive link. "
            "Otherwise, answer based on the context provided."
        )

        # 4. CONNECT TO PINECONE
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
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
        return {"response": "System maintenance. Give me a minute! 🤖"}

@app.get("/")
def home():
    return {"status": "Online", "message": "OPENAI BMSIT-BOT ACTIVE ✅"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
