from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

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
INDEX_NAME = "bmsit-chatbot"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ CRITICAL ERROR: API Keys missing from .env file!")

# 2. CONFIGURE AI (Globally)
print("⚙️ Setting up Gemini 2.0 Flash...")
try:
    embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    print("✅ Gemini 2.0 Configured Successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to configure Gemini: {e}")

# --- PASTE THIS RIGHT AFTER app = FastAPI() ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 This allows ANY website (Vercel, Localhost) to talk to it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------


# --- PERSONALITIES & INSTRUCTIONS ---
PROMPTS = {
    "Study Buddy": (
        "You are a helpful student peer. "
        "1. TIMETABLES: NEVER use Markdown tables or grids (like | Column |). They look broken. "
        "   - Instead, if the user asks for the full timetable, JUST give the [OFFICIAL DOCUMENT LINK]. "
        "   - If they ask for a specific day, use a simple list: 'Monday: 9am - Math'. "
        "2. VISUALS: Use Mermaid.js (graph TD) ONLY for concepts/processes, NOT for schedules. "
        "3. LINKS: Always prioritize the direct link for documents."
    ),
    "The Professor": (
        "You are a strict academic Professor. "
        "1. FORMATTING: Do NOT generate ASCII/Markdown tables. They are unprofessional. "
        "2. RESPONSE: For schedules, provide the [OFFICIAL DOCUMENT LINK] immediately. "
        "3. DETAILS: If specific class details are requested, state them in a clear sentence or bullet list."
    ),
    "The Bro": (
        "You are a chill friend. "
        "1. NO TABLES: Don't make those ugly grid tables. "
        "2. RESPONSE: If they want the timetable, just say 'Here is the link' and drop the [OFFICIAL DOCUMENT LINK]. "
        "3. SPECIFICS: Only list specific classes if they ask 'When is Physics?'."
    ),
    "ELI5": (
        "Explain simply. "
        "Never draw complex tables. Give the link if they need the schedule."
    )
}

class ChatRequest(BaseModel):
    message: str
    year: str
    mode: str = "Study Buddy"
    token: str = None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # 3. SELECT PERSONA
        system_instruction = PROMPTS.get(request.mode, PROMPTS["Study Buddy"])

        # 4. CONNECT TO DATABASE
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 5. CREATE CUSTOM PROMPT
        template_str = (
            f"{system_instruction}\n\n"
            "Context from notes/drive:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_template = PromptTemplate(template_str)

        # 6. FILTER BY YEAR
        filters = MetadataFilters(filters=[ExactMatchFilter(key="year", value=request.year)])
        
        # 7. QUERY
        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=filters,
            text_qa_template=qa_template
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}

    except Exception as e:
        print(f"❌ CRASH LOG: {e}")
        if "429" in str(e):
            return {"response": "I'm thinking too fast! Please wait 30 seconds. (Speed Limit Reached)"}
        return {"response": "I'm having trouble accessing my brain. Check the terminal for the error."}

@app.get("/")
def home():
    return {"status": "Active", "message": "BMSIT Chatbot Brain is Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)