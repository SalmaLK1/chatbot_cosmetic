from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# ==============================
# INITIALISATION DE LA BASE DE DONNÉES
# ==============================

# Instance SQLAlchemy pour la gestion de la base de données
# Sera initialisée avec l'application Flask via db.init_app(app)
db = SQLAlchemy()

# ==============================
# MODÈLE CHATTHREAD - GESTION DES FILS DE CONVERSATION
# ==============================

class ChatThread(db.Model):
    """
    Représente un fil de conversation entre un utilisateur et l'assistant.
    Chaque thread contient une série de messages et peut être archivé.
    """
    
    # Nom de la table dans la base de données
    __tablename__ = "chat_threads"
    
    # ==============================
    # COLONNES DE LA TABLE
    # ==============================
    
    # Identifiant unique du thread (clé primaire)
    id = db.Column(db.String(100), primary_key=True)  # thread_id
    
    # Identifiant de l'utilisateur propriétaire du thread
    user_id = db.Column(db.String(100), nullable=False)
    
    # Titre du thread, généré automatiquement ou défini par l'utilisateur
    title = db.Column(db.String(255), default="Nouveau chat")
    
    # Date et heure de création du thread
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Date et heure de la dernière mise à jour (automatique)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Statut d'archivage pour masquer sans supprimer
    archived = db.Column(db.Boolean, default=False)
    
    # ==============================
    # RELATIONS AVEC LES AUTRES TABLES
    # ==============================
    
    # Relation one-to-many avec les messages
    # cascade="all, delete-orphan" : suppression automatique des messages liés
    messages = db.relationship(
        "ChatMessage", 
        backref="thread",           # Crée une référence inverse depuis ChatMessage
        lazy=True,                  # Chargement paresseux des messages
        cascade="all, delete-orphan" # Supprime les messages si le thread est supprimé
    )
    
    # ==============================
    # MÉTHODES DE SÉRIALISATION
    # ==============================
    
    def to_dict(self):
        """
        Convertit l'objet thread en dictionnaire pour l'API.
        Inclut toutes les propriétés du thread.
        
        Returns:
            dict: Représentation JSON du thread
        """
        return {
            "thread_id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "archived": self.archived
        }
    
    def to_preview_dict(self):
        """
        Crée une version allégée du thread pour les listes.
        Inclut un aperçu du dernier message.
        
        Returns:
            dict: Aperçu du thread avec dernier message tronqué
        """
        # Récupération du dernier message s'il existe
        last_message = self.messages[-1].message if self.messages else ""
        
        # Troncature du message pour l'aperçu
        preview_message = (last_message[:50] + '...') if len(last_message) > 50 else last_message
        
        return {
            "thread_id": self.id,
            "title": self.title,
            "last_message": preview_message,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "archived": self.archived
        }

# ==============================
# MODÈLE CHATMESSAGE - GESTION DES MESSAGES
# ==============================

class ChatMessage(db.Model):
    """
    Représente un message individuel dans un fil de conversation.
    Peut être de l'utilisateur ou de l'assistant.
    """
    
    # Nom de la table dans la base de données
    __tablename__ = "history"
    
    # ==============================
    # COLONNES DE LA TABLE
    # ==============================
    
    # Identifiant unique auto-incrémenté (clé primaire)
    id = db.Column(db.Integer, primary_key=True)
    
    # Identifiant de l'utilisateur (redundant mais utile pour les requêtes directes)
    user_id = db.Column(db.String(100), nullable=False)
    
    # Identifiant de session pour le regroupement logique
    session_id = db.Column(db.String(100), nullable=False)
    
    # Clé étrangère vers la table chat_threads
    thread_id = db.Column(db.String(100), db.ForeignKey("chat_threads.id"), nullable=False)
    
    # Rôle de l'émetteur : "user" ou "assistant"
    role = db.Column(db.String(20), nullable=False)
    
    # Contenu textuel du message
    message = db.Column(db.Text, nullable=False)
    
    # Horodatage de création du message
    created_at = db.Column(db.DateTime, default=datetime.utcnow)