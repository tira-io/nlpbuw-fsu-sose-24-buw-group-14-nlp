from pathlib import Path
import json
import sys
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import nltk

# Add the src directory to the system path using an absolute path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from summarizer import summarize_article
from nltk.corpus import stopwords

nltk.download('stopwords')

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Generate summaries
    df['summary'] = df['story'].apply(lambda x: summarize_article(x, stop_words))
    df = df.drop(columns=['story']).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / 'predictions.jsonl', orient='records', lines=True)
