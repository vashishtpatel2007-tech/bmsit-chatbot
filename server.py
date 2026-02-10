import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "bmsit-chatbot")

# 1. CONFIGURE AI (STABLE 2026 VERSION)
try:
    # Must match the update_brain dimensionality
    embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-001", 
        api_key=GOOGLE_API_KEY,
        output_dimensionality=768
    )
    llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = llm
except Exception as e:
    print(f"❌ AI CONFIG ERROR: {e}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE = {"1": "...", "2": "...", "3": "...", "4": "..."}
PERSONAS = {"Study Buddy": "...", "The Professor": "...", "The Bro": "...", "ELI5": "..."}

class ChatRequest(BaseModel):
    message: str
    year: str = "1"
    mode: str = "Study Buddy"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        year = str(request.year) if str(request.year) in DATABASE else "1"
        persona = PERSONAS.get(request.mode, "Study Buddy")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        query_engine = index.as_query_engine(
            similarity_top_k=5, 
            filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=year)]),
            system_prompt=f"{persona}\nDrive: {DATABASE.get(year)}",
            embed_model=embed_model
        )
        
        response = query_engine.query(request.message)
        return {"response": str(response)}
    except Exception as e:
        print(f"❌ ERROR: {e}"); return {"response": "Brain glitch! Try again. 🤖"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
