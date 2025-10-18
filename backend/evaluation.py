import logging
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================
# LAZY LOADING DU CROSS-ENCODER
# ==============================

# Variable globale pour le chargement paresseux du Cross-Encoder
# Ce modèle est coûteux en mémoire, donc chargé seulement si nécessaire
_cross_encoder = None

def load_cross_encoder():
    """
    Charge le modèle Cross-Encoder de manière paresseuse.
    Le Cross-Encoder est utilisé pour le re-ranking de précision.
    
    Returns:
        CrossEncoder: Modèle Cross-Encoder initialisé
    """
    global _cross_encoder
    if _cross_encoder is None:
        # Chargement du modèle cross-encoder optimisé pour le re-ranking
        # "cross-encoder/ms-marco-MiniLM-L-6-v2" : modèle spécialisé MS MARCO
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logging.info("CrossEncoder loaded")
    return _cross_encoder

# ==============================
# RE-RANKING DES DOCUMENTS
# ==============================

def rerank_documents(query, docs, top_k=4, use_reranking=True):
    """
    Réordonne les documents par pertinence par rapport à la requête.
    Utilise un Cross-Encoder pour une évaluation de précision.
    
    Args:
        query (str): Requête de l'utilisateur
        docs (list): Liste de documents à réordonner
        top_k (int): Nombre de documents à retourner après re-ranking
        use_reranking (bool): Active/désactive le re-ranking
    
    Returns:
        list: Documents réordonnés par pertinence
    """
    # Vérification de la liste des documents
    if not docs:
        return []

    # Mode léger sans re-ranking - retourne les premiers documents
    if not use_reranking:
        return docs[:top_k]

    # Chargement du Cross-Encoder pour le re-ranking de précision
    cross_encoder = load_cross_encoder()
    
    # Création des paires (requête, document) pour l'évaluation
    pairs = [(query, doc.page_content) for doc in docs]
    
    # Calcul des scores de pertinence avec le Cross-Encoder
    # Le Cross-encoder évalue directement la pertinence requête-document
    scores = cross_encoder.predict(pairs)
    
    # Tri des documents par score décroissant
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    # Retour des top_k documents les plus pertinents
    return [doc for _, doc in scored_docs[:top_k]]

# ==============================
# ÉVALUATION DE LA QUALITÉ DES RÉPONSES
# ==============================

def evaluate_answer_quality(answer: str, context_docs: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Évalue la qualité d'une réponse en calculant sa similarité sémantique 
    avec les documents de contexte utilisés pour la génération.
    
    Args:
        answer (str): Réponse générée par le modèle
        context_docs (list): Liste des documents de contexte utilisés
        model_name (str): Modèle SentenceTransformer pour les embeddings
    
    Returns:
        float or None: Score de similarité moyen arrondi à 4 décimales, 
                      ou None en cas d'erreur
    """
    try:
        # Chargement du modèle SentenceTransformer pour les embeddings
        sbert_model = SentenceTransformer(model_name)
        
        # Encodage de la réponse en vecteur sémantique
        answer_embedding = sbert_model.encode([answer], convert_to_tensor=True)
        
        # Extraction du texte des documents de contexte
        doc_texts = [doc.page_content for doc in context_docs]
        
        # Encodage des documents de contexte en vecteurs sémantiques
        doc_embeddings = sbert_model.encode(doc_texts, convert_to_tensor=True)
        
        # Calcul des similarités cosinus entre la réponse et chaque document
        # La similarité cosinus mesure la proximité sémantique dans l'espace vectoriel
        similarities = cosine_similarity(answer_embedding, doc_embeddings)[0]
        
        # Calcul du score moyen de similarité
        mean_score = np.mean(similarities)
        
        # Retour du score arrondi à 4 décimales
        return round(mean_score, 4)
        
    except Exception as e:
        # Gestion robuste des erreurs d'évaluation
        logging.warning(f"Erreur évaluation qualité : {e}")
        return None