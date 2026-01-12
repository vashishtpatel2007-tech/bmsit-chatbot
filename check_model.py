import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load keys
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ No API Key found in .env file")
    exit()

# 2. Connect to Google
genai.configure(api_key=api_key)

print(f"🔑 Checking keys for: {api_key[:10]}...")
print("------------------------------------------------")

try:
    # 3. Ask Google for the list
    print("📡 Contacting Google Servers...")
    count = 0
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            count += 1
            
    if count == 0:
        print("⚠️ No Chat models found. Your API Key might be invalid or needs the 'Generative Language API' enabled in Google Cloud.")
        
except Exception as e:
    print(f"❌ Error connecting: {e}")