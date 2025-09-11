from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ChatThread(db.Model):
    __tablename__ = "chat_threads"
    id = db.Column(db.String(100), primary_key=True)  # thread_id
    user_id = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(255), default="Nouveau chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # ➕
    archived = db.Column(db.Boolean, default=False)  #  pour archiver/restaurer

    messages = db.relationship("ChatMessage", backref="thread", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "thread_id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "archived": self.archived
        }

    def to_preview_dict(self):
        """Renvoie un aperçu du thread avec le dernier message."""
        last_message = self.messages[-1].message if self.messages else ""
        return {
            "thread_id": self.id,
            "title": self.title,
            "last_message": (last_message[:50] + '...') if len(last_message) > 50 else last_message,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "archived": self.archived
        }

class ChatMessage(db.Model):
    __tablename__ = "history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    thread_id = db.Column(db.String(100), db.ForeignKey("chat_threads.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user / assistant
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
