from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
import threading
from datetime import datetime

# Import des modules backend personnalisés
from backend.backendtow import (
    process_question,
    handle_uploaded_file,
    handle_multiple_uploaded_files,
    rag_fusion_multi_docs,
    load_faiss_index,
    add_document_to_index,
    generate_export_file,
    embeddings,
)
from backend.models import db, ChatThread, ChatMessage
from backend.chat_service import handle_question, get_chat_history, generate_title_from_message
from langchain_community.vectorstores import FAISS
from backend.config import INDEX_PATH

# ==============================
# INITIALISATION DE L'APPLICATION FLASK
# ==============================

app = Flask(__name__)

# Configuration de l'application
app.config['UPLOAD_FOLDER'] = 'uploads'  # Dossier pour les fichiers uploadés
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'  # Base SQLite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Désactive le tracking des modifications

# Création du dossier d'upload s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialisation de la base de données avec l'application Flask
db.init_app(app)

# Configuration du système de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# CONFIGURATION ADMIN ET SÉCURITÉ
# ==============================

# Token d'administration - en production utiliser variables d'environnement
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "mon_token_secret")

# Verrou pour les opérations sur l'index FAISS (évite les accès concurrents)
index_lock = threading.Lock()

def check_admin_auth():
    """
    Vérifie l'authentification admin via token Bearer.
    
    Returns:
        bool: True si authentifié, False sinon
    """
    token = request.headers.get("Authorization", "")
    return token == f"Bearer {ADMIN_TOKEN}"

# ==============================
# FONCTIONS UTILITAIRES
# ==============================

def create_thread_if_not_exists(user_id, thread_id):
    """
    Crée un thread de conversation s'il n'existe pas déjà.
    
    Args:
        user_id (str): Identifiant de l'utilisateur
        thread_id (str): Identifiant du thread
    
    Returns:
        ChatThread: Thread existant ou nouvellement créé
    """
    thread = ChatThread.query.filter_by(id=thread_id).first()
    if not thread:
        new_thread = ChatThread(
            id=thread_id,
            user_id=user_id,
            title="Nouvelle conversation",
            created_at=datetime.utcnow()
        )
        db.session.add(new_thread)
        db.session.commit()
        return new_thread
    return thread

# ==============================
# ROUTES PRINCIPALES DE L'APPLICATION
# ==============================

@app.route("/")
def index():
    """
    Route racine - sert l'interface chat principale.
    
    Returns:
        Response: Page HTML du chat
    """
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Point d'entrée principal pour les questions et uploads de fichiers.
    Gère à la fois le RAG et le mode conversation simple.
    
    Returns:
        JSON: Réponse de l'assistant avec contexte
    """
    try:
        # Extraction des paramètres de la requête
        question = request.form.get("question", "").strip()
        use_rag = request.form.get("use_rag", "true").lower() == "true"
        session_id = request.form.get("session_id")
        user_id = request.form.get("user_id") or "anonymous"
        thread_id = request.form.get("thread_id")
        nb_messages = int(request.form.get("nb_messages", "3"))

        # Validation des paramètres requis
        if not session_id:
            return jsonify({"error": "session_id manquant"}), 400
        
        # Génération d'un thread_id si non fourni
        if not thread_id:
            thread_id = 'thread_' + os.urandom(8).hex()

        # Création du thread si nécessaire
        thread = create_thread_if_not_exists(user_id, thread_id)
        
        # Récupération des fichiers uploadés
        files = request.files.getlist("file")
        
        # Validation qu'il y a au moins une question ou des fichiers
        if not question and not files:
            return jsonify({"error": "Aucune question ni fichier reçu."}), 400

        logging.info(f"/ask reçu - user_id:{user_id} session_id:{session_id} thread_id:{thread_id}")

        # Initialisation des variables de réponse
        user_msg = question if question else ""
        answer = ""
        context = []

        # Section critique protégée par verrou (accès à l'index FAISS)
        with index_lock:
            if files:
                # Traitement avec fichiers uploadés
                answer = handle_multiple_uploaded_files(
                    files,
                    question=question,
                    chat_history=get_chat_history(user_id, session_id, thread_id, nb_messages),
                    use_rag=use_rag,
                    nb_messages=nb_messages
                )
                # Enrichissement du message utilisateur avec les noms de fichiers
                if question:
                    user_msg += " (Fichiers : " + ", ".join([f.filename for f in files]) + ")"
            else:
                # Traitement question seule
                chat_history = get_chat_history(user_id, session_id, thread_id, nb_messages)
                if use_rag:
                    # Mode RAG avec recherche documentaire
                    answer, context = rag_fusion_multi_docs(
                        query=question,
                        chat_history=chat_history,
                        nb_messages=nb_messages
                    )
                else:
                    # Mode conversation simple
                    answer = process_question(
                        question,
                        use_rag=False,
                        chat_history=chat_history,
                        nb_messages=nb_messages
                    )
                    context = []

        # Sauvegarde de l'échange en base de données
        handle_question(user_id, session_id, thread_id, user_msg, answer)

        # Mise à jour du titre du thread si c'est une nouvelle conversation
        if thread and (not thread.title or thread.title == "Nouvelle conversation"):
            thread.title = generate_title_from_message(user_msg)
            thread.created_at = thread.created_at or datetime.utcnow()
            db.session.commit()

        # Sérialisation du contexte pour la réponse JSON
        context_serializable = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in context]

        return jsonify({
            "answer": answer,
            "context": context_serializable,
            "session_id": session_id,
            "thread_id": thread_id
        })

    except Exception as e:
        logging.error(f"Erreur serveur /ask : {e}", exc_info=True)
        return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500

@app.route("/export", methods=["POST"])
def export():
    """
    Exporte une conversation dans différents formats (TXT, DOCX, PDF).
    
    Returns:
        File: Fichier d'export à télécharger
    """
    try:
        data = request.get_json()
        # Validation des données requises
        if not data or "answer" not in data or "context" not in data or "format" not in data:
            return jsonify({"error": "Données incomplètes pour l'export."}), 400
        
        # Génération du fichier d'export
        file_path = generate_export_file(data, format=data["format"])
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logging.error(f"Erreur export fichier : {e}", exc_info=True)
        return jsonify({"error": f"Erreur export : {str(e)}"}), 500

@app.route("/history", methods=["GET"])
def history():
    """
    Récupère l'historique complet d'un thread de conversation.
    
    Returns:
        JSON: Liste des messages du thread
    """
    session_id = request.args.get("session_id", "default")
    user_id = request.args.get("user_id", "anonymous")
    thread_id = request.args.get("thread_id")
    
    if not thread_id:
        return jsonify({"error": "thread_id manquant"}), 400
        
    try:
        messages = get_chat_history(user_id, session_id, thread_id, nb_messages=100)
        # Transformation en format linéaire pour l'affichage
        result = []
        for m in messages:
            result.append({"role": "user", "message": m['user']})
            result.append({"role": "assistant", "message": m['assistant']})
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Erreur récupération historique complet : {e}", exc_info=True)
        return jsonify({"error": "Erreur récupération historique"}), 500

@app.route("/chats", methods=["GET"])
def list_chats():
    """
    Liste tous les threads de conversation d'un utilisateur.
    
    Returns:
        JSON: Liste des threads avec métadonnées
    """
    user_id = request.args.get("user_id", "anonymous")
    try:
        threads = ChatThread.query.filter_by(user_id=user_id).order_by(ChatThread.created_at.desc()).all()
        return jsonify([t.to_dict() for t in threads])
    except Exception as e:
        logging.error(f"Erreur récupération des threads : {e}", exc_info=True)
        return jsonify({"error": "Erreur récupération des discussions"}), 500

@app.route("/threads", methods=["POST"])
def create_thread():
    """
    Crée un nouveau thread de conversation.
    
    Returns:
        JSON: Détails du thread créé
    """
    data = request.get_json() or {}
    user_id = data.get("user_id", "anonymous")
    thread_id = 'thread_' + os.urandom(8).hex()
    try:
        new_thread = ChatThread(
            id=thread_id, 
            user_id=user_id, 
            title="Nouvelle conversation", 
            created_at=datetime.utcnow()
        )
        db.session.add(new_thread)
        db.session.commit()
        return jsonify(new_thread.to_dict())
    except Exception as e:
        logging.error(f"Erreur création thread : {e}", exc_info=True)
        return jsonify({"error": "Impossible de créer une nouvelle conversation."}), 500

@app.route("/threads/<thread_id>", methods=["PUT"])
def rename_thread(thread_id):
    """
    Renomme un thread existant.
    
    Args:
        thread_id (str): Identifiant du thread à renommer
    
    Returns:
        JSON: Confirmation de la mise à jour
    """
    data = request.get_json()
    if not data or "title" not in data:
        return jsonify({"error": "Titre manquant"}), 400
        
    new_title = data["title"].strip()
    if not new_title:
        return jsonify({"error": "Titre vide"}), 400
        
    thread = ChatThread.query.filter_by(id=thread_id).first()
    if not thread:
        return jsonify({"error": "Thread introuvable"}), 404
        
    thread.title = new_title
    db.session.commit()
    return jsonify({"message": "Titre mis à jour"})

@app.route("/threads/<thread_id>", methods=["DELETE"])
def delete_thread(thread_id):
    """
    Supprime un thread et tous ses messages.
    
    Args:
        thread_id (str): Identifiant du thread à supprimer
    
    Returns:
        JSON: Confirmation de la suppression
    """
    thread = ChatThread.query.filter_by(id=thread_id).first()
    if not thread:
        return jsonify({"error": "Thread introuvable"}), 404
        
    try:
        # Suppression en cascade des messages puis du thread
        ChatMessage.query.filter_by(thread_id=thread_id).delete()
        db.session.delete(thread)
        db.session.commit()
        return jsonify({"message": "Thread supprimé"})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Erreur suppression thread : {e}", exc_info=True)
        return jsonify({"error": f"Erreur lors de la suppression: {str(e)}"}), 500

# ==============================
# ROUTES ADMINISTRATION
# ==============================

@app.route("/admin/reload_index", methods=["POST"])
def admin_reload_index():
    """
    Recharge l'index FAISS (admin seulement).
    
    Returns:
        JSON: Statut de l'opération
    """
    if not check_admin_auth():
        return jsonify({"error": "Unauthorized"}), 401
        
    with index_lock:
        try:
            load_faiss_index()
            return jsonify({"status": "Index FAISS rechargé."})
        except Exception as e:
            logging.error(f"Erreur reload index : {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

@app.route("/admin/reset_index", methods=["POST"])
def admin_reset_index():
    """
    Réinitialise complètement l'index FAISS (admin seulement).
    
    Returns:
        JSON: Statut de l'opération
    """
    if not check_admin_auth():
        return jsonify({"error": "Unauthorized"}), 401
        
    with index_lock:
        try:
            # Création d'un nouvel index vide
            db_faiss = FAISS.from_documents([], embeddings)
            db_faiss.save_local(INDEX_PATH)
            
            # Mise à jour de la référence dans le module backend
            from backend import backendtow as backend_mod
            backend_mod.db = db_faiss
            
            return jsonify({"status": "Index FAISS réinitialisé."})
        except Exception as e:
            logging.error(f"Erreur reset index : {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

@app.route("/admin/add_document", methods=["POST"])
def admin_add_document():
    """
    Ajoute un document directement à l'index (admin seulement).
    
    Returns:
        JSON: Statut de l'opération
    """
    if not check_admin_auth():
        return jsonify({"error": "Unauthorized"}), 401
        
    with index_lock:
        try:
            text = request.json.get("text", "")
            metadata = request.json.get("metadata", {})
            
            if not text.strip():
                return jsonify({"error": "Texte vide fourni."}), 400
                
            success = add_document_to_index(text, metadata)
            if success:
                return jsonify({"status": "Document ajouté à l'index."})
            else:
                return jsonify({"error": "Échec de l'ajout du document."}), 500
                
        except Exception as e:
            logging.error(f"Erreur add document : {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

# ==============================
# POINT D'ENTRÉE DE L'APPLICATION
# ==============================

if __name__ == "__main__":
    # Création des tables de base de données
    with app.app_context():
        db.create_all()
    
    # Chargement de l'index FAISS au démarrage
    load_faiss_index()
    
    # Lancement de l'application Flask
    app.run(debug=True, use_reloader=False)