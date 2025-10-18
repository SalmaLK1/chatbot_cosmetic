import os

# ==============================
# CONFIGURATION DES CHEMINS DU SYSTÈME
# ==============================

# Répertoire de base de l'application
# os.getcwd() retourne le répertoire de travail courant
BASE_DIR = os.getcwd()

# Chemin vers l'index FAISS pour la recherche vectorielle
# L'index est stocké dans un sous-dossier 'index/arx_faiss'
INDEX_PATH = os.path.join(BASE_DIR, "index/arx_faiss")

# Répertoire pour le cache des fichiers temporaires
# Stocke les transcriptions et textes extraits pour éviter les retraitements
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ==============================
# CRÉATION DES RÉPERTOIRES SI INEXISTANTS
# ==============================

# Création du répertoire cache avec exist_ok=True pour éviter les erreurs si existe déjà
# Mode 0o755 : permissions read/write/execute pour owner, read/execute pour group et others
os.makedirs(CACHE_DIR, exist_ok=True)

# Création du répertoire parent de l'index FAISS
# os.path.dirname() extrait le chemin du dossier parent (index/)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# ==============================
# CONFIGURATION DES CLÉS API ET LIMITES
# ==============================

# Clé API Google pour les services Google (non utilisée dans le code actuel mais configurée)
# NOTE: En production, cette clé devrait être dans les variables d'environnement
GOOGLE_API_KEY = ''

# Limite de taille maximale pour les fichiers uploadés (en Mégaoctets)
# Empêche l'upload de fichiers trop volumineux qui pourraient saturer la mémoire
MAX_FILE_SIZE_MB = 20  # Limite taille fichier en Mo (ajustable)