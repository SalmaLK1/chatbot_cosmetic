from backend.models import ChatMessage, ChatThread, db
from sqlalchemy import desc
from datetime import datetime

class DocumentTooLargeError(Exception):
    pass


def generate_title_from_message(message: str, max_words: int = 6) -> str:
    """
    Génère un titre court à partir de la première ligne du message utilisateur.
    """
    title = message.strip().split("\n")[0]
    words = title.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def save_chat_turn(user_id: str, session_id: str, thread_id: str, role: str, message: str) -> None:
    """
    Sauvegarde un message dans la base de données.
    """
    if not message.strip():
        return  # Ignore les messages vides
    try:
        chat_message = ChatMessage(
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            role=role,
            message=message,
            created_at=datetime.utcnow()
        )
        db.session.add(chat_message)
    except Exception:
        db.session.rollback()
        raise


def get_chat_history(user_id: str, session_id: str, thread_id: str, nb_messages: int = 3) -> list[dict]:
    """
    Récupère les derniers échanges (nb_messages * 2) entre l'utilisateur et l'assistant.
    Format : [{"user": ..., "assistant": ...}, ...]
    """
    messages = ChatMessage.query.filter_by(
        user_id=user_id,
        session_id=session_id,
        thread_id=thread_id
    ).order_by(desc(ChatMessage.id)).limit(nb_messages * 2).all()

    messages.reverse()  # ordre chronologique

    chat_history = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            chat_history.append({
                "user": messages[i].message,
                "assistant": messages[i + 1].message
            })
    return chat_history


def get_thread_or_create(user_id: str, thread_id: str, question: str) -> ChatThread:
    """
    Récupère un thread existant ou le crée si inexistant, avec titre dynamique.
    """
    thread = ChatThread.query.filter_by(id=thread_id).first()
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


def handle_question(user_id: str, session_id: str, thread_id: str, question: str, answer: str) -> None:
    """
    Gère un tour complet de conversation : crée le thread si nécessaire,
    enregistre la question et la réponse.
    """
    try:
        thread = get_thread_or_create(user_id, thread_id, question)
        save_chat_turn(user_id, session_id, thread.id, "user", question)
        save_chat_turn(user_id, session_id, thread.id, "assistant", answer)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
