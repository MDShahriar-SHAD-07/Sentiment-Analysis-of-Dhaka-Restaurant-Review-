import os
import shutil
import logging
import pandas as pd
from config import get_settings
from data_loader import initialize_data
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

def build_local_vector_db():
    """
    Rebuilds the vector database locally using the processed data.
    This bypasses any specific environment issues (like on Kaggle).
    """
    print("🚀 Starting Local Vector DB Build...")
    
    # 1. Load Data
    if not settings.PROCESSED_DATA_PATH.exists():
        pickle_path = settings.DATA_DIR / "processed_reviews.pkl"
        if pickle_path.exists():
             df = pd.read_pickle(pickle_path)
             print(f"✅ Loaded {len(df)} reviews from pickle")
        else:
            print("❌ processed_reviews.csv or .pkl not found in 'data/' folder.")
            print("   Please download 'data.zip' from Kaggle and extract it to 'data/'")
            return
    else:
        df = pd.read_csv(settings.PROCESSED_DATA_PATH)
        print(f"✅ Loaded {len(df)} reviews from CSV")

    # 2. Initialize Embeddings
    print(f"📥 Loading Embedding Model ({settings.EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # 3. Create Documents
    print("📄 Creating Documents...")
    documents = []
    for _, row in df.iterrows():
        # Reconstruct aspect summary
        aspect_summary = {}
        for aspect in settings.ASPECTS.keys():
            if row.get(f'{aspect}_mentioned', False):
                aspect_summary[aspect] = row.get(f'{aspect}_sentiment')

        # Handle potential string/float type mismatches
        business_name = str(row['business_name'])
        try:
             business_name_norm = str(row['business_name_normalized'])
        except:
             business_name_norm = business_name.lower().strip()

        doc = Document(
            page_content=str(row['review_text']),
            metadata={
                'restaurant': business_name,
                'restaurant_normalized': business_name_norm,
                'rating': float(row['review_rating']),
                'sentiment': str(row.get('overall_sentiment', 'neutral')),
                'confidence': float(row.get('sentiment_confidence', 0.0)),
                'aspects': json.dumps(aspect_summary),
                'conflict': str(row.get('rating_sentiment_conflict', 'No Conflict')),
                'has_conflict': bool(row.get('has_conflict', False))
            }
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents")

    # 4. Build Vector DB
    db_path = str(settings.VECTOR_DB_PATH)
    if os.path.exists(db_path):
        print(f"⚠️ Removing existing DB at {db_path}...")
        shutil.rmtree(db_path)
    
    print(f"🏗️ Building Vector Store at {db_path}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
        client_settings=Settings(anonymized_telemetry=False)
    )
    
    # Force persist if needed
    try:
        vector_store.persist()
    except:
        pass
        
    print("🎉 Success! Vector Database built locally.")
    print("   You can now run 'python main.py'")

if __name__ == "__main__":
    build_local_vector_db()
