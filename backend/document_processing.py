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

# Lazy loading whisper and blip
_whisper_model = None
_blip_processor = None
_blip_model = None

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
        logging.info("Whisper model loaded")
    return _whisper_model

def load_blip_model():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        logging.info("BLIP model loaded")
    return _blip_processor, _blip_model

def describe_image_with_blip(image_path_or_file):
    try:
        blip_processor, blip_model = load_blip_model()
        if isinstance(image_path_or_file, str):
            image = Image.open(image_path_or_file).convert("RGB")
        else:
            image_path_or_file.seek(0)
            image = Image.open(image_path_or_file).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Erreur lors de la description de l'image : {e}")
        return f"[Erreur lors de la description de l'image : {e}]"

def chunk_text_semantically(text, max_tokens=500, overlap_tokens=100, tokenizer=None):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_tokens = len(tokenizer.encode(sentence)) if tokenizer else len(sentence.split())

        if current_len + sent_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            # Overlap sémantique
            overlap = []
            token_sum = 0
            for sent in reversed(current_chunk):
                token_sum += len(tokenizer.encode(sent)) if tokenizer else len(sent.split())
                overlap.insert(0, sent)
                if token_sum >= overlap_tokens:
                    break
            current_chunk = overlap
            current_len = sum(len(tokenizer.encode(s)) if tokenizer else len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def transcribe_audio(audio_input):
    try:
        whisper_model = load_whisper_model()
        import tempfile
        file_bytes = audio_input.read()
        audio_input.seek(0)
        file_hash = hashlib.md5(file_bytes).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}_asr.json")
        if os.path.exists(cache_path):
            logging.info("Chargement transcription en cache")
            return json.load(open(cache_path, "r", encoding="utf-8"))["text"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path)
        os.remove(tmp_path)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result["text"]
    except Exception as e:
        logging.error(f"Erreur transcription : {e}")
        return f"Erreur transcription : {e}"

def extract_text(file, max_file_size_mb=MAX_FILE_SIZE_MB):
    try:
        file_bytes = file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            return f"Erreur : fichier trop volumineux ({file_size_mb:.2f} Mo). Limite : {max_file_size_mb} Mo."
        file.seek(0)
        file_hash = hashlib.md5(file_bytes).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}_text.json")
        if os.path.exists(cache_path):
            logging.info("Chargement texte extrait en cache")
            return json.load(open(cache_path, "r", encoding="utf-8"))["text"]

        ext = os.path.splitext(file.filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(file.filename)
        text = ""

        if ext == ".pdf":
            text = "\n".join([p.extract_text() or "" for p in PdfReader(file).pages])
        elif ext == ".docx":
            text = "\n".join([p.text for p in DocxDocument(file).paragraphs])
        elif ext in [".png", ".jpg", ".jpeg"]:
            file.seek(0)
            text = describe_image_with_blip(file)
        elif "audio" in (mime_type or "") or ext in [".mp3", ".wav", ".m4a"]:
            text = transcribe_audio(file)
        else:
            file.seek(0)
            text = file.read().decode("utf-8", errors="ignore")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False, indent=2)

        file.seek(0)
        return text
    except Exception as e:
        logging.error(f"Erreur extraction : {e}")
        return f"Erreur extraction : {e}"
