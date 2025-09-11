import logging
import time
import os
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tiktoken import get_encoding
from .config import INDEX_PATH
from .evaluation import rerank_documents
from .document_processing import chunk_text_semantically, extract_text
from .file_utils import get_title_from_filename, get_file_hash
from .models import db, ChatMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ==============================
# Config modèle MPT
# ==============================
MODEL_NAME = "mosaicml/mpt-7b-instruct"  # ou "mosaicml/mpt-3b-instruct" si RAM limitée

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",            # compatible Windows CPU
    torch_dtype=torch.float32,   # float32 pour CPU
    low_cpu_mem_usage=True
)

# ==============================
# Embeddings & FAISS
# ==============================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_faiss = None

def load_faiss_index():
    global db_faiss
    try:
        db_faiss = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logging.info("✅ Index FAISS chargé.")
    except Exception as e:
        logging.warning(f"⚠️ Index non trouvé ou invalide, création d'un index vide : {e}")
        db_faiss = FAISS.from_documents([], embeddings)
        db_faiss.save_local(INDEX_PATH)
        logging.info("✅ Index FAISS vide créé.")

def reset_faiss_index():
    global db_faiss
    logging.info("⚠️ Réinitialisation de l’index FAISS...")
    db_faiss = FAISS.from_documents([], embeddings)
    db_faiss.save_local(INDEX_PATH)
    logging.info("✅ Index FAISS réinitialisé.")

def get_existing_document_ids():
    try:
        return set(
            doc.metadata.get("document_id")
            for doc in db_faiss.similarity_search("", k=1000)
            if doc.metadata.get("document_id")
        )
    except Exception as e:
        logging.warning(f"Erreur récupération des document_id : {e}")
        return set()

def add_document_to_index(text, metadata=None):
    global db_faiss
    try:
        if not text.strip():
            logging.warning("Texte vide, rien à indexer.")
            return False

        document_id = metadata.get("document_id") if metadata else None
        existing_ids = get_existing_document_ids()
        if document_id and document_id in existing_ids:
            logging.info(f"📛 Document déjà indexé : {document_id}")
            return False

        chunks = chunk_text_semantically(text, max_tokens=500, overlap_tokens=100)
        docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_length": len(chunk),
                "title": metadata.get("title", "Sans titre") if metadata else "Sans titre"
            })
            docs.append(Document(page_content=chunk, metadata=chunk_metadata))

        db_faiss.add_documents(docs)
        db_faiss.save_local(INDEX_PATH)
        logging.info(f"✅ {len(docs)} chunks ajoutés avec métadonnées enrichies.")
        return True
    except Exception as e:
        logging.error(f"Erreur ajout document à l'index : {e}")
        return False

# ==============================
# Historique depuis BDD
# ==============================
def get_chat_history_from_db(user_id, session_id, thread_id, limit=10):
    try:
        messages = (
            ChatMessage.query
            .filter_by(user_id=user_id, session_id=session_id, thread_id=thread_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit * 2)
            .all()
        )
        messages.reverse()

        history_pairs = []
        current_pair = {}
        for msg in messages:
            if msg.role == "user":
                current_pair["user"] = msg.message
            elif msg.role == "assistant":
                current_pair["assistant"] = msg.message

            if "user" in current_pair and "assistant" in current_pair:
                history_pairs.append(current_pair)
                current_pair = {}

        return history_pairs
    except Exception as e:
        logging.error(f"Erreur récupération historique : {e}")
        return []

# ==============================
# Appel modèle MPT
# ==============================
def call_mpt(prompt, max_tokens=512):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Erreur MPT : {e}")
        return " Réponse impossible."

# ==============================
# RAG Fusion
# ==============================
def rag_fusion_multi_docs(query, chat_history=None, k=6, nb_messages=10, use_reranking=True):
    if db_faiss is None:
        logging.error("Index FAISS non chargé")
        return " Index non chargé.", []

    if chat_history is None:
        chat_history = []

    try:
        retrieved_docs = db_faiss.similarity_search(query, k=k)
        docs = rerank_documents(query, retrieved_docs, top_k=k, use_reranking=use_reranking)
    except Exception as e:
        logging.error(f"Erreur recherche documentaire : {e}")
        return f"Erreur recherche documentaire : {e}", []

    tokenizer_gpt = get_encoding("gpt2")
    context_text = ""
    context_docs = []

    for doc in docs:
        context_text += f"[Source: {doc.metadata.get('title', 'Document inconnu')}]\n{doc.page_content}\n\n"
        context_docs.append(doc)

    summarized_history = ""
    for turn in chat_history[-nb_messages:]:
        summarized_history += f"Utilisateur : {turn['user']}\nAssistant : {turn['assistant']}\n"

    prompt = f"""
Tu es un assistant IA expert.
Question :
{query}

Contexte documentaire :
{context_text}

Historique résumé :
{summarized_history}

Réponse :
""".strip()

    full_answer = call_mpt(prompt)
    return full_answer, context_docs

# ==============================
# Direct prompt (sans RAG)
# ==============================
def rag_direct_prompt(query, chat_history=None, nb_messages=10):
    if chat_history is None:
        chat_history = []

    summarized_history = ""
    for turn in chat_history[-nb_messages:]:
        summarized_history += f"Utilisateur : {turn['user']}\nAssistant : {turn['assistant']}\n"

    prompt = f"""
Question :
{query}

Historique résumé :
{summarized_history}

Réponse :
""".strip()

    return call_mpt(prompt)

# ==============================
# Process Question + Historique
# ==============================
def process_question(user_id, session_id, thread_id, question, use_rag=True, nb_messages=10, use_reranking=True):
    chat_history = get_chat_history_from_db(user_id, session_id, thread_id, limit=nb_messages)

    if use_rag:
        answer, _ = rag_fusion_multi_docs(question, chat_history, nb_messages=nb_messages, use_reranking=use_reranking)
    else:
        answer = rag_direct_prompt(question, chat_history, nb_messages=nb_messages)

    try:
        db.session.add(ChatMessage(user_id=user_id, session_id=session_id, thread_id=thread_id, role="user", message=question))
        db.session.add(ChatMessage(user_id=user_id, session_id=session_id, thread_id=thread_id, role="assistant", message=answer))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.error(f"Erreur enregistrement historique : {e}")

    return answer

# ==============================
# Gestion fichiers
# ==============================
def handle_uploaded_file(file, user_id, session_id, thread_id, question=None, use_rag=True, nb_messages=10, use_reranking=True):
    text = extract_text(file)
    if not text or "Erreur" in text:
        return text

    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    doc_id = get_file_hash(file_bytes)
    title = get_title_from_filename(file.filename)

    metadata = {
        "document_id": doc_id,
        "source": file.filename,
        "title": title
    }

    add_document_to_index(text, metadata=metadata)

    if question:
        return process_question(user_id, session_id, thread_id, question, use_rag, nb_messages, use_reranking)
    return " Fichier indexé avec succès."
