import os
from dotenv import load_dotenv
from llama_index.readers.google import GoogleDriveReader

# 1. SETUP
load_dotenv()

# PASTE THE FOLDER ID WHERE THE TIMETABLE IS (Year 3 based on your screenshot)
FOLDER_ID = "1fIZRxNrGmz5BwzNjbCsLdHfOHRK9MU1e" 

def check_file_content():
    print("🔍 Connecting to Drive to inspect files...")
    
    try:
        # Load the files
        loader = GoogleDriveReader(service_account_key_path="credentials.json")
        docs = loader.load_data(folder_id=FOLDER_ID)
        
        found_it = False
        
        for doc in docs:
            # Check if this is the timetable file (checks filename inside metadata)
            filename = doc.metadata.get('file_name', 'Unknown')
            
            # We look for "Physics" or "Timetable" in the filename
            if "Physics" in filename or "Timetable" in filename:
                found_it = True
                print(f"\n📄 FOUND FILE: {filename}")
                print("------------------------------------------------")
                
                # PRINT WHAT THE BOT SEES
                content = doc.text.strip()
                if not content:
                    print("⚠️ CRITICAL ISSUE: The text is EMPTY!")
                    print("   (This means the PDF is an image/scan. The bot cannot read it.)")
                else:
                    print(f"✅ Text Content Preview (First 500 chars):\n{content[:500]}")
                print("------------------------------------------------")

        if not found_it:
            print("❌ Could not find a file with 'Physics' or 'Timetable' in the name.")
            print("   Here are the files I found:")
            for doc in docs:
                print(f"   - {doc.metadata.get('file_name')}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_file_content()