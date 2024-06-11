import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences
