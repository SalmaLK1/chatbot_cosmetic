import logging
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Lazy loading CrossEncoder for reranking (costly)
_cross_encoder = None

def load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logging.info("CrossEncoder loaded")
    return _cross_encoder

def rerank_documents(query, docs, top_k=4, use_reranking=True):
    if not docs:
        return []

    if not use_reranking:
        # Mode light, sans reranking, retourne docs tels quels (ou premier k)
        return docs[:top_k]

    cross_encoder = load_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def evaluate_answer_quality(answer: str, context_docs: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        sbert_model = SentenceTransformer(model_name)
        answer_embedding = sbert_model.encode([answer], convert_to_tensor=True)
        doc_texts = [doc.page_content for doc in context_docs]
        doc_embeddings = sbert_model.encode(doc_texts, convert_to_tensor=True)
        similarities = cosine_similarity(answer_embedding, doc_embeddings)[0]
        mean_score = np.mean(similarities)
        return round(mean_score, 4)
    except Exception as e:
        logging.warning(f"Erreur évaluation qualité : {e}")
        return None
