import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
print("📢 SANITY CHECK: I AM RUNNING THE NEW CODE WITH EMBEDDING-001")

# --- IMPORTS ---
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

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ CRITICAL ERROR: API Keys missing from .env file!")

# ==============================================================================
# 🗄️ MASTER FOLDER DATABASE (Your Real Links)
# ==============================================================================
DATABASE = {
    "1": "https://drive.google.com/drive/folders/1Yv-tfstUnQytvhvdLP02j6IDiolovIWI?usp=drive_link",
    "2": "https://drive.google.com/drive/folders/1gGPWHjZSF0Z22fus_yRrX_aq3zKws5Bp?usp=drive_link",
    "3": "https://drive.google.com/drive/folders/1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e?usp=drive_link",
    "4": "https://drive.google.com/drive/folders/17Ga5lrRQ-d8aLEOhZ24qZ7vWL8bXUpY1?usp=drive_link"
}

# 2. CONFIGURE AI
print("⚙️ Setting up Gemini 2.0...")
try:
    # FIXED: Switched to 'embedding-001' to fix 404 error
    embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = llm
except Exception as e:
    raise RuntimeError(f"❌ Failed to configure Gemini: {e}")

# 3. INITIALIZE APP
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ✨ PERSONA RULES (REFINED)
# ==============================================================================
PERSONA_RULES = {
    "Study Buddy": (
        "You are 'Alex', an energetic, super-supportive BMSIT senior. "
        "VIBE: Positive, encouraging, high energy! Use emojis like 🚀, ✨, 📚. "
        "GOAL: Make the student feel capable. "
        "If asked for a file, say 'I got you covered! Here's the stash:'"
    ),
    "The Professor": (
        "You are Professor Sharma, a distinguished academic. "
        "VIBE: Helpful, professional, precise, and polite. "
        "GOAL: Provide high-quality information. "
        "FORMATTING: Use bullet points or numbered lists if requested. "
        "Do not use slang. Ensure answers are complete and accurate."
    ),
    "The Bro": (
        "You are 'Sam', the chillest guy on campus. "
        "VIBE: Casual, uses slang (fam, easy scene, bet, dw). "
        "GOAL: Give the answer instantly. "
        "If they want a file, say 'Say less, here's the link:'"
    ),
    "ELI5": (
        "You are a patient Tutor. "
        "VIBE: Gentle, slow, and clear. "
        "GOAL: Explain hard concepts simply using analogies. No jargon."
    )
}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"
    token: str = None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # 1. GET THE MASTER LINK
        selected_year = str(request.year)
        if selected_year not in DATABASE: selected_year = "1"
        master_folder_link = DATABASE[selected_year]

        # 2. CONSTRUCT SYSTEM PROMPT
        persona_instruction = PERSONA_RULES.get(request.mode, PERSONA_RULES["Study Buddy"])
        
        base_instruction = (
            f"{persona_instruction}\n\n"
            f"You are assisting a Year {selected_year} student.\n"
            f"OFFICIAL DRIVE LINK: {master_folder_link}\n\n"
            "🧠 LOGIC PROTOCOL:\n"
            "1. FILE REQUESTS (e.g., 'timetable', 'syllabus', 'notes', 'pdf'):\n"
            "   - You ONLY have the Master Folder. Reply: 'Here is the Master Folder: {master_folder_link}'\n"
            "2. KNOWLEDGE QUESTIONS (e.g., 'when is exams?', 'explain unit 1'):\n"
            "   - Search the 'Context' below. Answer directly.\n"
            "   - IF the user asks to format (e.g., 'pointwise'), DO IT.\n"
        )

        # 3. CONNECT TO BRAIN
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 4. CREATE PROMPT
        template_str = (
            f"{base_instruction}\n\n"
            "CONTEXT FROM YOUR BRAIN:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "USER SAYS: {query_str}\n"
            "YOUR REPLY:"
        )
        qa_template = PromptTemplate(template_str)

        # 5. QUERY
        filters = MetadataFilters(filters=[ExactMatchFilter(key="year", value=request.year)])
        
        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=filters,
            text_qa_template=qa_template
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}

    except Exception as e:
        print(f"❌ CRASH LOG: {e}")
        return {"response": "My brain is having a hiccup! Please try again in a sec. 🤖"}

@app.get("/")
def home():
    return {"status": "Active", "message": "BMSIT Vibe Check Passed ✅"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
