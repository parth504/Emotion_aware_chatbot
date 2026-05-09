import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from datasets import load_dataset
from langchain_core.documents import Document

def build_vector_db():
    print("1. Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./mentalchat_chroma_db"
    
    print("2. Downloading ShenLab/MentalChat16K dataset...")
    dataset = load_dataset("ShenLab/MentalChat16K")
    
    # Extract the training data split
    try:
        train_data = dataset['train']
    except Exception:
        train_data = list(dataset.values())[0]

    docs = []
    limit = 10000 
    
    print(f"3. Formatting {limit} conversations for the Vector Database...")
    for i, row in enumerate(train_data):
        if i >= limit:
            break
            
        inst = row.get('instruction', '')
        inp = row.get('input', '')
        out = row.get('output', '')
        
        # Structure the text for maximum contextual retrieval
        content = f"Patient: {inst} {inp}\nTherapist: {out}"
        docs.append(Document(page_content=content.strip(), metadata={"source": "MentalChat16K", "id": i}))
    
    print("4. Building ChromaDB. This will take a few minutes...")
    
    # Ingest the documents in batches to prevent SQLite/Memory overload
    batch_size = 2000
    vectorstore = None
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"   -> Embedding batch {i // batch_size + 1} of {(len(docs) - 1) // batch_size + 1}...")
        
        if vectorstore is None:
            # Initialize the database with the first batch
            vectorstore = Chroma.from_documents(
                documents=batch, 
                embedding=embeddings, 
                persist_directory=persist_directory
            )
        else:
            # Add subsequent batches to the existing database
            vectorstore.add_documents(documents=batch)
            
    print(f"\n✅ Success! Database fully built and saved to: {persist_directory}")

if __name__ == "__main__":
    build_vector_db()