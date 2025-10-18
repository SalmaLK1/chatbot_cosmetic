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
# CONFIGURATION DU MODÈLE MPT
# ==============================

# Modèle MPT-7B de MosaicML, version optimisée pour le dialogue
# Alternative: "mosaicml/mpt-3b-instruct" pour machines avec RAM limitée
MODEL_NAME = "mosaicml/mpt-7b-instruct"

# Initialisation du tokenizer pour le prétraitement du texte
# Gère la tokenization des prompts et des réponses
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Chargement du modèle principal avec configuration optimisée CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",            # Spécifique Windows/CPU - pas de support GPU
    torch_dtype=torch.float32,   # Précision simple pour stabilité sur CPU
    low_cpu_mem_usage=True       # Réduction consommation mémoire lors du chargement
)

# ==============================
# GESTION DES EMBEDDINGS ET BASE VECTORIELLE FAISS
# ==============================

# Modèle d'embedding SentenceTransformers - équilibre performance/vitesse
# all-MiniLM-L6-v2: 384 dimensions, rapide et efficace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Variable globale pour l'index FAISS - initialisée à None au démarrage
db_faiss = None

def load_faiss_index():
    """
    Charge l'index FAISS depuis le stockage local.
    Crée un index vide si aucun index existant n'est trouvé.
    
    Returns:
        None
    """
    global db_faiss
    try:
        # Tentative de chargement de l'index existant
        # allow_dangerous_deserialization=True nécessaire pour FAISS mais nécessite confiance dans la source
        db_faiss = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logging.info("✅ Index FAISS chargé.")
    except Exception as e:
        # Fallback: création d'un nouvel index vide
        logging.warning(f"⚠️ Index non trouvé ou invalide, création d'un index vide : {e}")
        db_faiss = FAISS.from_documents([], embeddings)  # Index vide avec modèle d'embedding
        db_faiss.save_local(INDEX_PATH)  # Sauvegarde pour usage futur
        logging.info("✅ Index FAISS vide créé.")

def reset_faiss_index():
    """
    Réinitialise complètement l'index FAISS.
    Utile pour les tests ou la maintenance.
    
    Returns:
        None
    """
    global db_faiss
    logging.info("⚠️ Réinitialisation de l'index FAISS...")
    # Création d'un nouvel index vide
    db_faiss = FAISS.from_documents([], embeddings)
    db_faiss.save_local(INDEX_PATH)  # Persistance immédiate
    logging.info("✅ Index FAISS réinitialisé.")

def get_existing_document_ids():
    """
    Récupère tous les IDs de documents présents dans l'index.
    Utilisé pour éviter les doublons lors de l'indexation.
    
    Returns:
        set: Ensemble des IDs de documents existants
    """
    try:
        # Recherche vide pour récupérer tous les documents (limité à 1000)
        return set(
            doc.metadata.get("document_id")
            for doc in db_faiss.similarity_search("", k=1000)
            if doc.metadata.get("document_id")  # Filtre les documents sans ID
        )
    except Exception as e:
        logging.warning(f"Erreur récupération des document_id : {e}")
        return set()  # Retourne un set vide en cas d'erreur

def add_document_to_index(text, metadata=None):
    """
    Ajoute un document à l'index FAISS après traitement et découpage.
    
    Args:
        text (str): Texte du document à indexer
        metadata (dict, optional): Métadonnées associées au document
    
    Returns:
        bool: True si l'ajout réussi, False sinon
    """
    global db_faiss
    try:
        # Validation du texte d'entrée
        if not text.strip():
            logging.warning("Texte vide, rien à indexer.")
            return False

        # Extraction et vérification de l'ID du document
        document_id = metadata.get("document_id") if metadata else None
        existing_ids = get_existing_document_ids()
        
        # Vérification de doublon
        if document_id and document_id in existing_ids:
            logging.info(f"📛 Document déjà indexé : {document_id}")
            return False

        # Découpage sémantique du texte en chunks
        # max_tokens=500: taille optimale pour la recherche
        # overlap_tokens=100: préservation du contexte entre chunks
        chunks = chunk_text_semantically(text, max_tokens=500, overlap_tokens=100)
        docs = []
        
        # Création des objets Document avec métadonnées enrichies
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i,  # Position du chunk dans le document
                "chunk_length": len(chunk),  # Longueur en caractères
                "title": metadata.get("title", "Sans titre") if metadata else "Sans titre"
            })
            docs.append(Document(page_content=chunk, metadata=chunk_metadata))

        # Ajout à l'index et sauvegarde
        db_faiss.add_documents(docs)
        db_faiss.save_local(INDEX_PATH)
        logging.info(f"✅ {len(docs)} chunks ajoutés avec métadonnées enrichies.")
        return True
        
    except Exception as e:
        logging.error(f"Erreur ajout document à l'index : {e}")
        return False

# ==============================
# GESTION DE L'HISTORIQUE DEPUIS LA BASE DE DONNÉES
# ==============================

def get_chat_history_from_db(user_id, session_id, thread_id, limit=10):
    """
    Récupère l'historique des conversations depuis la base de données.
    Reconstruit les paires question/réponse pour le contexte.
    
    Args:
        user_id: Identifiant de l'utilisateur
        session_id: Identifiant de session
        thread_id: Identifiant du thread de conversation
        limit (int): Nombre de paires de messages à récupérer
    
    Returns:
        list: Liste de dictionnaires contenant les paires user/assistant
    """
    try:
        # Récupération des messages depuis la BDD
        messages = (
            ChatMessage.query
            .filter_by(user_id=user_id, session_id=session_id, thread_id=thread_id)
            .order_by(ChatMessage.created_at.desc())  # Plus récents en premier
            .limit(limit * 2)  # ×2 car on veut des paires
            .all()
        )
        messages.reverse()  # Remise dans l'ordre chronologique

        # Reconstruction des paires question/réponse
        history_pairs = []
        current_pair = {}
        
        for msg in messages:
            if msg.role == "user":
                current_pair["user"] = msg.message
            elif msg.role == "assistant":
                current_pair["assistant"] = msg.message

            # Paire complète détectée
            if "user" in current_pair and "assistant" in current_pair:
                history_pairs.append(current_pair)
                current_pair = {}  # Réinitialisation pour la paire suivante

        return history_pairs
        
    except Exception as e:
        logging.error(f"Erreur récupération historique : {e}")
        return []  # Retourne liste vide en cas d'erreur

# ==============================
# APPEL DU MODÈLE MPT POUR LA GÉNÉRATION
# ==============================

def call_mpt(prompt, max_tokens=512):
    """
    Exécute le modèle MPT avec le prompt fourni.
    
    Args:
        prompt (str): Texte d'entrée pour le modèle
        max_tokens (int): Nombre maximum de tokens à générer
    
    Returns:
        str: Réponse générée par le modèle
    """
    try:
        # Tokenization du prompt
        inputs = tokenizer(prompt, return_tensors="pt")  # Tenseurs PyTorch
        
        # Génération de la réponse
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Décodage et nettoyage de la réponse
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        logging.error(f"Erreur MPT : {e}")
        return " Réponse impossible."  # Message d'erreur générique

# ==============================
# SYSTÈME RAG FUSION AVANCÉ
# ==============================

def rag_fusion_multi_docs(query, chat_history=None, k=6, nb_messages=10, use_reranking=True):
    """
    Implémente le système RAG Fusion avec recherche multi-documents.
    Combine recherche vectorielle, re-ranking et contexte historique.
    
    Args:
        query (str): Question de l'utilisateur
        chat_history (list): Historique des conversations
        k (int): Nombre de documents à récupérer
        nb_messages (int): Nombre de messages d'historique à inclure
        use_reranking (bool): Active/désactive le re-ranking
    
    Returns:
        tuple: (réponse, documents de contexte)
    """
    # Vérification de l'initialisation de FAISS
    if db_faiss is None:
        logging.error("Index FAISS non chargé")
        return " Index non chargé.", []

    # Initialisation par défaut de l'historique
    if chat_history is None:
        chat_history = []

    try:
        # Étape 1: Recherche vectorielle initiale
        retrieved_docs = db_faiss.similarity_search(query, k=k)
        
        # Étape 2: Re-ranking des résultats (optionnel)
        docs = rerank_documents(query, retrieved_docs, top_k=k, use_reranking=use_reranking)
        
    except Exception as e:
        logging.error(f"Erreur recherche documentaire : {e}")
        return f"Erreur recherche documentaire : {e}", []

    # Initialisation du tokenizer pour comptage (non utilisé actuellement)
    tokenizer_gpt = get_encoding("gpt2")
    
    # Construction du contexte documentaire
    context_text = ""
    context_docs = []

    for doc in docs:
        # Formatage des sources avec métadonnées
        context_text += f"[Source: {doc.metadata.get('title', 'Document inconnu')}]\n{doc.page_content}\n\n"
        context_docs.append(doc)

    # Construction de l'historique résumé
    summarized_history = ""
    for turn in chat_history[-nb_messages:]:  # Derniers N messages seulement
        summarized_history += f"Utilisateur : {turn['user']}\nAssistant : {turn['assistant']}\n"

    # Construction du prompt final structuré
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

    # Appel au modèle pour génération
    full_answer = call_mpt(prompt)
    return full_answer, context_docs

# ==============================
# MODE PROMPT DIRECT (SANS RAG)
# ==============================

def rag_direct_prompt(query, chat_history=None, nb_messages=10):
    """
    Mode sans RAG - utilise seulement l'historique et la question.
    Utile pour les questions générales ne nécessitant pas de documentation.
    
    Args:
        query (str): Question de l'utilisateur
        chat_history (list): Historique des conversations
        nb_messages (int): Nombre de messages d'historique à inclure
    
    Returns:
        str: Réponse générée par le modèle
    """
    # Initialisation par défaut
    if chat_history is None:
        chat_history = []

    # Construction de l'historique résumé
    summarized_history = ""
    for turn in chat_history[-nb_messages:]:
        summarized_history += f"Utilisateur : {turn['user']}\nAssistant : {turn['assistant']}\n"

    # Prompt simplifié sans contexte documentaire
    prompt = f"""
Question :
{query}

Historique résumé :
{summarized_history}

Réponse :
""".strip()

    return call_mpt(prompt)

# ==============================
# TRAITEMENT PRINCIPAL DES QUESTIONS
# ==============================

def process_question(user_id, session_id, thread_id, question, use_rag=True, nb_messages=10, use_reranking=True):
    """
    Point d'entrée principal pour le traitement des questions.
    Gère le mode RAG/direct et la persistance en base.
    
    Args:
        user_id: Identifiant de l'utilisateur
        session_id: Identifiant de session
        thread_id: Identifiant du thread
        question (str): Question à traiter
        use_rag (bool): Active/désactive le mode RAG
        nb_messages (int): Nombre de messages d'historique
        use_reranking (bool): Active/désactive le re-ranking
    
    Returns:
        str: Réponse générée
    """
    # Récupération de l'historique depuis la BDD
    chat_history = get_chat_history_from_db(user_id, session_id, thread_id, limit=nb_messages)

    # Sélection du mode de traitement
    if use_rag:
        answer, _ = rag_fusion_multi_docs(question, chat_history, nb_messages=nb_messages, use_reranking=use_reranking)
    else:
        answer = rag_direct_prompt(question, chat_history, nb_messages=nb_messages)

    # Persistance de l'échange en base de données
    try:
        # Enregistrement de la question utilisateur
        db.session.add(ChatMessage(
            user_id=user_id, 
            session_id=session_id, 
            thread_id=thread_id, 
            role="user", 
            message=question
        ))
        
        # Enregistrement de la réponse de l'assistant
        db.session.add(ChatMessage(
            user_id=user_id, 
            session_id=session_id, 
            thread_id=thread_id, 
            role="assistant", 
            message=answer
        ))
        
        db.session.commit()  # Validation de la transaction
        
    except Exception as e:
        db.session.rollback()  # Annulation en cas d'erreur
        logging.error(f"Erreur enregistrement historique : {e}")

    return answer

# ==============================
# GESTION DES FICHIERS UPLOADÉS
# ==============================

def handle_uploaded_file(file, user_id, session_id, thread_id, question=None, use_rag=True, nb_messages=10, use_reranking=True):
    """
    Traite un fichier uploadé: extraction, indexation et réponse optionnelle.
    
    Args:
        file: Objet fichier uploadé
        user_id: Identifiant de l'utilisateur
        session_id: Identifiant de session
        thread_id: Identifiant du thread
        question (str, optional): Question associée au fichier
        use_rag (bool): Active/désactive le mode RAG
        nb_messages (int): Nombre de messages d'historique
        use_reranking (bool): Active/désactive le re-ranking
    
    Returns:
        str: Message de confirmation ou réponse à la question
    """
    # Extraction du texte depuis le fichier
    text = extract_text(file)
    
    # Vérification de la réussite de l'extraction
    if not text or "Erreur" in text:
        return text  # Retourne l'erreur d'extraction

    # Préparation pour le hachage du fichier
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)  # Reset pour usage futur
    
    # Génération d'ID unique basé sur le contenu
    doc_id = get_file_hash(file_bytes)
    title = get_title_from_filename(file.filename)

    # Métadonnées pour l'indexation
    metadata = {
        "document_id": doc_id,  # ID unique pour déduplication
        "source": file.filename,  # Nom original du fichier
        "title": title  # Titre extrait du nom de fichier
    }

    # Indexation du document
    add_document_to_index(text, metadata=metadata)

    # Réponse à une question si fournie
    if question:
        return process_question(user_id, session_id, thread_id, question, use_rag, nb_messages, use_reranking)
    
    return " Fichier indexé avec succès."  # Message de confirmation simple