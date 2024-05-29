from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compute_cosine_similarity(df: pd.DataFrame):
    vectorizer = TfidfVectorizer()
    similarities = []
    for _, row in df.iterrows():
        sentences = [row['sentence1'], row['sentence2']]
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarities.append(similarity[0][0])
    return similarities
