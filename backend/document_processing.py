import os
import json
import logging
import mimetypes
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
from nltk.tokenize import sent_tokenize
from .config import CACHE_DIR, MAX_FILE_SIZE_MB
import hashlib

# ==============================
# LAZY LOADING DES MODÈLES LOURDS
# ==============================

# Variables globales pour le chargement paresseux des modèles
# Évite de charger les modèles tant qu'ils ne sont pas nécessaires
_whisper_model = None  # Modèle de transcription audio
_blip_processor = None  # Processeur pour le modèle BLIP
_blip_model = None     # Modèle de description d'images

def load_whisper_model():
    """
    Charge le modèle Whisper pour la transcription audio de manière paresseuse.
    Le modèle n'est chargé qu'à la première utilisation.
    
    Returns:
        whisper.Model: Modèle Whisper initialisé
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper  # Import local pour éviter dépendance inutile
        # Chargement du modèle "base" - bon compromis performance/précision
        _whisper_model = whisper.load_model("base")
        logging.info("Whisper model loaded")
    return _whisper_model

def load_blip_model():
    """
    Charge le modèle BLIP pour la génération de légendes d'images de manière paresseuse.
    
    Returns:
        tuple: (BlipProcessor, BlipForConditionalGeneration) - processeur et modèle
    """
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        # Modèle pré-entraîné pour la génération de légendes
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        logging.info("BLIP model loaded")
    return _blip_processor, _blip_model

# ==============================
# DESCRIPTION D'IMAGES AVEC BLIP
# ==============================

def describe_image_with_blip(image_path_or_file):
    """
    Génère une description textuelle d'une image using le modèle BLIP.
    
    Args:
        image_path_or_file (str or file-like): Chemin vers l'image ou objet fichier
    
    Returns:
        str: Description textuelle de l'image ou message d'erreur
    """
    try:
        # Chargement paresseux des modèles BLIP
        blip_processor, blip_model = load_blip_model()
        
        # Gestion des différents types d'entrée (chemin ou fichier)
        if isinstance(image_path_or_file, str):
            # Ouverture depuis un chemin de fichier
            image = Image.open(image_path_or_file).convert("RGB")
        else:
            # Ouverture depuis un objet fichier - reset du curseur
            image_path_or_file.seek(0)
            image = Image.open(image_path_or_file).convert("RGB")
        
        # Prétraitement de l'image pour le modèle
        inputs = blip_processor(images=image, return_tensors="pt")
        
        # Génération de la légende
        out = blip_model.generate(**inputs)
        
        # Décodage de la sortie tokenisée en texte lisible
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        logging.error(f"Erreur lors de la description de l'image : {e}")
        return f"[Erreur lors de la description de l'image : {e}]"

# ==============================
# DÉCOUPAGE SÉMANTIQUE DU TEXTE
# ==============================

def chunk_text_semantically(text, max_tokens=500, overlap_tokens=100, tokenizer=None):
    """
    Découpe un texte en chunks sémantiques basés sur les phrases.
    Préserve la cohérence sémantique avec un chevauchement contrôlé.
    
    Args:
        text (str): Texte à découper
        max_tokens (int): Nombre maximum de tokens par chunk
        overlap_tokens (int): Nombre de tokens de chevauchement entre chunks
        tokenizer: Tokenizer pour compter les tokens (optionnel)
    
    Returns:
        list: Liste des chunks de texte
    """
    # Tokenization en phrases avec NLTK
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        # Calcul du nombre de tokens dans la phrase
        sent_tokens = len(tokenizer.encode(sentence)) if tokenizer else len(sentence.split())

        # Vérification si l'ajout dépasse la limite
        if current_len + sent_tokens > max_tokens:
            # Sauvegarde du chunk courant
            chunks.append(" ".join(current_chunk))
            
            # Calcul du chevauchement sémantique
            overlap = []
            token_sum = 0
            # Parcours inversé pour prendre les dernières phrases
            for sent in reversed(current_chunk):
                token_sum += len(tokenizer.encode(sent)) if tokenizer else len(sent.split())
                overlap.insert(0, sent)  # Insertion au début pour préserver l'ordre
                if token_sum >= overlap_tokens:
                    break
            
            # Nouveau chunk commence avec le chevauchement
            current_chunk = overlap
            current_len = sum(len(tokenizer.encode(s)) if tokenizer else len(s.split()) for s in current_chunk)

        # Ajout de la phrase au chunk courant
        current_chunk.append(sentence)
        current_len += sent_tokens

    # Ajout du dernier chunk s'il n'est pas vide
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ==============================
# TRANSCRIPTION AUDIO AVEC WHISPER
# ==============================

def transcribe_audio(audio_input):
    """
    Transcrit un fichier audio en texte using Whisper.
    Utilise un système de cache pour éviter les retranscriptions.
    
    Args:
        audio_input: Fichier audio à transcrire
    
    Returns:
        str: Texte transcrit ou message d'erreur
    """
    try:
        # Chargement paresseux du modèle Whisper
        whisper_model = load_whisper_model()
        
        import tempfile
        
        # Lecture et hash du fichier pour le cache
        file_bytes = audio_input.read()
        audio_input.seek(0)  # Reset pour usage futur
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Vérification du cache
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}_asr.json")
        if os.path.exists(cache_path):
            logging.info("Chargement transcription en cache")
            return json.load(open(cache_path, "r", encoding="utf-8"))["text"]
        
        # Création d'un fichier temporaire pour Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        # Transcription avec Whisper
        result = whisper_model.transcribe(tmp_path)
        
        # Nettoyage du fichier temporaire
        os.remove(tmp_path)
        
        # Sauvegarde dans le cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result["text"]
        
    except Exception as e:
        logging.error(f"Erreur transcription : {e}")
        return f"Erreur transcription : {e}"

# ==============================
# EXTRACTION DE TEXTE MULTI-FORMATS
# ==============================

def extract_text(file, max_file_size_mb=MAX_FILE_SIZE_MB):
    """
    Extrait le texte d'un fichier dans différents formats.
    Supporte PDF, DOCX, images, audio et fichiers texte.
    
    Args:
        file: Fichier à traiter
        max_file_size_mb (int): Taille maximale autorisée en Mo
    
    Returns:
        str: Texte extrait ou message d'erreur
    """
    try:
        # Lecture complète du fichier pour vérification taille
        file_bytes = file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        # Vérification de la taille du fichier
        if file_size_mb > max_file_size_mb:
            return f"Erreur : fichier trop volumineux ({file_size_mb:.2f} Mo). Limite : {max_file_size_mb} Mo."
        
        # Reset du curseur pour les traitements suivants
        file.seek(0)
        
        # Génération d'hash pour le système de cache
        file_hash = hashlib.md5(file_bytes).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}_text.json")
        
        # Vérification du cache
        if os.path.exists(cache_path):
            logging.info("Chargement texte extrait en cache")
            return json.load(open(cache_path, "r", encoding="utf-8"))["text"]

        # Analyse de l'extension et type MIME
        ext = os.path.splitext(file.filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(file.filename)
        text = ""

        # Traitement selon le type de fichier
        if ext == ".pdf":
            # Extraction texte depuis PDF
            text = "\n".join([p.extract_text() or "" for p in PdfReader(file).pages])
            
        elif ext == ".docx":
            # Extraction texte depuis DOCX
            text = "\n".join([p.text for p in DocxDocument(file).paragraphs])
            
        elif ext in [".png", ".jpg", ".jpeg"]:
            # Description d'image avec BLIP
            file.seek(0)
            text = describe_image_with_blip(file)
            
        elif "audio" in (mime_type or "") or ext in [".mp3", ".wav", ".m4a"]:
            # Transcription audio avec Whisper
            text = transcribe_audio(file)
            
        else:
            # Traitement par défaut pour les fichiers texte
            file.seek(0)
            text = file.read().decode("utf-8", errors="ignore")

        # Sauvegarde dans le cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False, indent=2)

        # Reset final du curseur
        file.seek(0)
        return text
        
    except Exception as e:
        logging.error(f"Erreur extraction : {e}")
        return f"Erreur extraction : {e}"