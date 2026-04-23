import os
import sys
import chromadb

# Add project root to path
sys.path.append(os.path.abspath("."))

def dump_reflections():
    db_path = os.path.join("checkpoints", "warmup", "chroma")
    if not os.path.exists(db_path):
        print(f"❌ DB path not found: {db_path}")
        return

    print(f"🔍 Opening ChromaDB at {db_path}...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="agent")
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return

    # Query for reflections
    results = collection.get(
        where={"layer": "reflection"},
        limit=10
    )

    if not results["ids"]:
        print("📭 No reflections found in the database layer='reflection'.")
        return

    print(f"✅ Found {len(results['ids'])} reflections:\n")
    for i in range(len(results["ids"])):
        meta = results['metadatas'][i]
        print(f"[{meta.get('date')}] {meta.get('symbol')}: {results['documents'][i]}")

if __name__ == "__main__":
    dump_reflections()
