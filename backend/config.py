import os

BASE_DIR = os.getcwd()
INDEX_PATH = os.path.join(BASE_DIR, "index/arx_faiss")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

GOOGLE_API_KEY = 'AIzaSyA-f4kIOE8E7dQaM2G611M5PyuSmiSdOkQ'
MAX_FILE_SIZE_MB = 20  # Limite taille fichier en Mo (ajustable)
