from backend.models import ChatMessage, ChatThread, db
from sqlalchemy import desc
from datetime import datetime

# ==============================
# EXCEPTIONS PERSONNALISÉES
# ==============================

class DocumentTooLargeError(Exception):
    """
    Exception levée lorsqu'un document dépasse la taille maximale autorisée.
    Permet une gestion spécifique des erreurs de taille de fichiers.
    """
    pass

# ==============================
# GÉNÉRATION DE TITRES
# ==============================

def generate_title_from_message(message: str, max_words: int = 6) -> str:
    """
    Génère un titre court et significatif à partir du premier message utilisateur.
    Utilise la première ligne et limite le nombre de mots pour la concision.
    
    Args:
        message (str): Message complet de l'utilisateur
        max_words (int): Nombre maximum de mots dans le titre
    
    Returns:
        str: Titre généré avec éventuellement des points de suspension si tronqué
    """
    # Extraction de la première ligne du message
    title = message.strip().split("\n")[0]
    
    # Découpage en mots et limitation
    words = title.split()
    
    # Reconstruction du titre avec limite de mots
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# ==============================
# GESTION DES MESSAGES
# ==============================

def save_chat_turn(user_id: str, session_id: str, thread_id: str, role: str, message: str) -> None:
    """
    Sauvegarde un message individuel dans la base de données.
    Ignore les messages vides pour optimiser le stockage.
    
    Args:
        user_id (str): Identifiant unique de l'utilisateur
        session_id (str): Identifiant de session navigateur
        thread_id (str): Identifiant du thread de conversation
        role (str): Rôle de l'émetteur ("user" ou "assistant")
        message (str): Contenu du message à sauvegarder
    
    Raises:
        Exception: En cas d'erreur lors de la sauvegarde, rollback automatique
    """
    # Validation : ignore les messages vides ou ne contenant que des espaces
    if not message.strip():
        return  # Sortie silencieuse pour les messages vides
    
    try:
        # Création de l'objet message avec horodatage actuel
        chat_message = ChatMessage(
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            role=role,
            message=message,
            created_at=datetime.utcnow()  # Horodatage UTC pour la cohérence
        )
        
        # Ajout à la session pour persistence
        db.session.add(chat_message)
        
    except Exception:
        # Rollback en cas d'erreur pour maintenir la cohérence des données
        db.session.rollback()
        raise  # Propagation de l'exception pour gestion supérieure

# ==============================
# RÉCUPÉRATION DE L'HISTORIQUE
# ==============================

def get_chat_history(user_id: str, session_id: str, thread_id: str, nb_messages: int = 3) -> list[dict]:
    """
    Récupère les derniers échanges de conversation sous forme de paires question/réponse.
    Format de retour adapté pour le contexte des modèles de langage.
    
    Args:
        user_id (str): Identifiant unique de l'utilisateur
        session_id (str): Identifiant de session navigateur
        thread_id (str): Identifiant du thread de conversation
        nb_messages (int): Nombre de paires question/réponse à récupérer
    
    Returns:
        list[dict]: Liste de dictionnaires au format {"user": "...", "assistant": "..."}
    """
    # Récupération des messages les plus récents
    messages = ChatMessage.query.filter_by(
        user_id=user_id,
        session_id=session_id,
        thread_id=thread_id
    ).order_by(desc(ChatMessage.id)).limit(nb_messages * 2).all()   # Tri par ID décroissant (plus récent en premier)
     # ×2 car on veut des paires
    

    # Remise en ordre chronologique (plus ancien en premier)
    messages.reverse()

    # Reconstruction des paires question/réponse
    chat_history = []
    for i in range(0, len(messages) - 1, 2):
        # Vérification de l'alternance user/assistant
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            chat_history.append({
                "user": messages[i].message,
                "assistant": messages[i + 1].message
            })
    
    return chat_history

# ==============================
# GESTION DES THREADS DE CONVERSATION
# ==============================

def get_thread_or_create(user_id: str, thread_id: str, question: str) -> ChatThread:
    """
    Récupère un thread existant ou le crée avec un titre généré automatiquement.
    Pattern commun de "get or create" pour la gestion des conversations.
    
    Args:
        user_id (str): Identifiant unique de l'utilisateur
        thread_id (str): Identifiant du thread à récupérer ou créer
        question (str): Première question pour générer le titre
    
    Returns:
        ChatThread: Instance du thread existant ou nouvellement créé
    """
    # Recherche du thread existant
    thread = ChatThread.query.filter_by(id=thread_id).first()
    
    # Création si non existant
    if not thread:
        title = generate_title_from_message(question)
        thread = ChatThread(
            id=thread_id,
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow()
        )
        db.session.add(thread)
    
    return thread

# ==============================
# GESTION COMPLÈTE D'UN ÉCHANGE
# ==============================

def handle_question(user_id: str, session_id: str, thread_id: str, question: str, answer: str) -> None:
    """
    Gère un échange complet de conversation : création du thread si nécessaire,
    sauvegarde de la question et de la réponse en une transaction atomique.
    
    Args:
        user_id (str): Identifiant unique de l'utilisateur
        session_id (str): Identifiant de session navigateur
        thread_id (str): Identifiant du thread de conversation
        question (str): Question posée par l'utilisateur
        answer (str): Réponse générée par l'assistant
    
    Raises:
        Exception: En cas d'erreur, rollback complet de la transaction
    """
    try:
        # Récupération ou création du thread avec titre généré
        thread = get_thread_or_create(user_id, thread_id, question)
        
        # Sauvegarde des deux messages (question et réponse)
        save_chat_turn(user_id, session_id, thread.id, "user", question)
        save_chat_turn(user_id, session_id, thread.id, "assistant", answer)
        
        # Validation de la transaction
        db.session.commit()
        
    except Exception:
        # Annulation de toute la transaction en cas d'erreur
        db.session.rollback()
        raise  # Propagation pour gestion supérieure