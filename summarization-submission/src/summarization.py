from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import networkx as nx
import nltk

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

def generate_summary(text, num_sentences=2):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Build the similarity graph
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences based on TextRank scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top ranked sentences to form the summary
    summary_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    return " ".join(summary_sentences)
