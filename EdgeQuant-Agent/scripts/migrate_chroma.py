import os
import chromadb
from datetime import datetime

# Path to the ChromaDB
DB_PATH = "checkpoints/warmup/chroma"

def migrate():
    print(f"🚀 Starting migration for ChromaDB at {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database path not found at {DB_PATH}")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    
    # We only have one collection named 'agent'
    try:
        collection = client.get_collection("agent")
    except Exception as e:
        print(f"❌ Error: Could not find collection 'agent'. {e}")
        return

    # Fetch all documents
    print("📊 Fetching all memories...")
    all_data = collection.get()
    
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    
    if not ids:
        print("ℹ️ No memories found in database.")
        return

    print(f"🔄 Migrating {len(ids)} memories...")
    
    updated_metadatas = []
    for meta in metadatas:
        # Extract existing ISO date string "2025-01-01"
        date_str = meta.get("date")
        if date_str:
            # Convert to YYYYMMDD integer
            date_int = int(date_str.replace("-", ""))
            meta["date_int"] = date_int
        updated_metadatas.append(meta)

    # Perform batch update
    # Note: collection.update requires ids and new metadatas
    collection.update(
        ids=ids,
        metadatas=updated_metadatas
    )

    print("✅ Migration complete! All memories now have 'date_int'.")

if __name__ == "__main__":
    migrate()
