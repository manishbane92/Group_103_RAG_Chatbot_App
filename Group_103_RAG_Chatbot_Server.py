# -------------------------------
# 1. Data Extraction and Preprocessing
# -------------------------------

import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF and splits it into meaningful chunks.
    """
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Split by double newlines to form paragraphs
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if len(para.strip()) > 100:  # simple filter for meaningful content
                    text_chunks.append(para.strip())
    return text_chunks

pdf_path = "JPMorgan Chase Bank, N.A. 2024 Annual Consolidated Financial Statements - Final.pdf"
chunks = extract_text_from_pdf(pdf_path)

# -------------------------------
# 2. Embedding-based Retrieval Setup
# -------------------------------

# Load an open-source embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype(np.float32)

# Build a FAISS index for dense embeddings
dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(doc_embeddings)

# -------------------------------
# BM25 for Keyword-Based Retrieval
# -------------------------------

# Tokenize chunks for BM25 (simple whitespace tokenization)
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# -------------------------------
# 4. Hybrid Retrieval Function (BM25 + Dense Embeddings)
# -------------------------------

def hybrid_search(query, alpha=0.5, initial_k=5, final_k=3):
    # BM25 scores for the full corpus
    tokenized_query = query.lower().split()
    bm25_all_scores = bm25.get_scores(tokenized_query)  # shape (len(chunks),)
    
    # Dense retrieval using FAISS for initial_k documents
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, initial_k)
    
    # Restrict BM25 scores to the retrieved subset
    bm25_subset = bm25_all_scores[indices[0]]
    
    # Compute dense cosine similarity for each retrieved document
    dense_scores = []
    for idx in indices[0]:
        doc_emb = doc_embeddings[idx]
        query_emb = query_embedding[0]
        cosine_sim = np.dot(doc_emb, query_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(query_emb))
        dense_scores.append(cosine_sim)
    
    # Normalize both BM25 and dense scores
    scaler = MinMaxScaler()
    bm25_norm = scaler.fit_transform(bm25_subset.reshape(-1, 1)).flatten()
    dense_norm = scaler.fit_transform(np.array(dense_scores).reshape(-1, 1)).flatten()
    
    # Combine scores (weighted sum)
    combined_scores = alpha * bm25_norm + (1 - alpha) * dense_norm
    
    # Rank the retrieved documents based on the combined scores
    ranked_indices = np.argsort(combined_scores)[::-1]
    ranked_chunks = [chunks[indices[0][i]] for i in ranked_indices]
    ranked_scores = [combined_scores[i] for i in ranked_indices]
    
    return ranked_chunks[:final_k], list(zip([chunks[indices[0][i]] for i in ranked_indices], ranked_scores))

# -------------------------------
# 5. Cross-Encoder Re-Ranking for Advanced RAG
# -------------------------------

# Load cross-encoder model for re-ranking
cross_encoder_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_score(query, text):
    inputs = cross_encoder_tokenizer(query, text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = cross_encoder_model(**inputs)
    score = outputs.logits.item()  # higher score indicates better relevance
    return score

def re_rank_with_cross_encoder(query, chunks_list):
    scored = []
    for text in chunks_list:
        score = cross_encoder_score(query, text)
        scored.append((text, score))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted

def advanced_retrieve(query, alpha=0.5, initial_k=5, final_k=3):
    # First, perform hybrid retrieval
    hybrid_chunks, hybrid_ranked = hybrid_search(query, alpha=alpha, initial_k=initial_k, final_k=initial_k)
    # Then, re-rank these chunks using the cross-encoder
    cross_ranked = re_rank_with_cross_encoder(query, hybrid_chunks)
    top_chunks = [text for text, score in cross_ranked[:final_k]]
    return top_chunks, cross_ranked

# -------------------------------
# 6. Response Generation using TinyLlama
# -------------------------------

# Initialize TinyLlama using a pipeline for text-generation
# Note: Adjust max_length, temperature, etc., as needed.
llm_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def generate_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Based on the following excerpts from JPMorgan Chase's 2024 financial statements:
{context}

Answer the question: {query}
"""
    return prompt

def generate_answer(prompt):
    # Use the TinyLlama pipeline for text generation
    # You may adjust max_length and other generation parameters as required.
    result = llm_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

# -------------------------------
# 7. Guardrail Implementation
# -------------------------------

def guardrail_input(query):
    # Input-side guardrail: block irrelevant/harmful queries.
    forbidden_terms = ["profanity", "inappropriate"]  # Extend as needed.
    for term in forbidden_terms:
        if term in query.lower():
            return False
    return True

def guardrail_output(response):
    # Output-side guardrail: check for placeholders or signs of hallucination.
    suspicious_phrases = ["placeholder", "generated answer", "N/A"]
    for phrase in suspicious_phrases:
        if phrase in response.lower():
            return False
    return True

def ask_with_guardrail(query):
    if not guardrail_input(query):
        return "Your query contains disallowed content.", None, None
    # Advanced retrieval
    top_chunks, cross_ranked = advanced_retrieve(query)
    prompt = generate_prompt(query, top_chunks)
    answer = generate_answer(prompt)
    # Apply output guardrail
    if not guardrail_output(answer):
        return "The generated answer was flagged as potentially misleading. Please try rephrasing your query.", top_chunks, cross_ranked
    return answer, top_chunks, cross_ranked
