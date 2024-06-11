import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from preprocess import preprocess_text

def compute_tfidf(sentences, stop_words):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer

def build_textrank_graph(tfidf_matrix):
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

def rank_sentences(sentences, tfidf_scores, textrank_scores):
    combined_scores = {}
    for idx, sentence in enumerate(sentences):
        combined_scores[sentence] = tfidf_scores[idx] + textrank_scores[idx]
    ranked_sentences = sorted(combined_scores, key=combined_scores.get, reverse=True)
    return ranked_sentences

def summarize_article(article, stop_words):
    sentences = preprocess_text(article)
    tfidf_matrix, vectorizer = compute_tfidf(sentences, stop_words)
    tfidf_scores = tfidf_matrix.sum(axis=1).A1
    textrank_scores = build_textrank_graph(tfidf_matrix)
    ranked_sentences = rank_sentences(sentences, tfidf_scores, textrank_scores)
    summary = " ".join(ranked_sentences[:3])  # Select top 3 sentences for summary
    return summary
