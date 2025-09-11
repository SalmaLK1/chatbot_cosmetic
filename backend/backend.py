# backend/backend.py

from .rag_engine import (
    load_faiss_index,
    handle_uploaded_file,
    handle_multiple_uploaded_files,
    process_question,
    reset_faiss_index,
    rag_fusion_multi_docs,
    add_document_to_index,
    
)
from .config import GOOGLE_API_KEY
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print(f"API Key utilisée: {GOOGLE_API_KEY}")

genai.configure(api_key=GOOGLE_API_KEY)

def init():
    load_faiss_index()
    
    logging.info("Backend initialisé.")
