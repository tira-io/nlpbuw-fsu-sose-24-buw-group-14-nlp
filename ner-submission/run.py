from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import spacy
import pandas as pd
import re

# Load the spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Define simple regex-based NER
def simple_ner(sentence):
    # Example regex patterns for different entity types
    patterns = {
        'PERSON': r'\b[A-Z][a-z]*\s[A-Z][a-z]*\b',
        'ORG': r'\b[A-Z][A-Za-z]*\s(?:Corporation|Inc|Company|Ltd|LLC|Group)\b',
        'GPE': r'\b[A-Z][a-z]*(?:\s[A-Z][a-z]*)*\b'
    }
    
    # Tokenize the sentence
    doc = nlp(sentence)
    
    # Initialize tags
    tags = ["O"] * len(doc)
    
    # Apply patterns to extract entities
    for ent_type, pattern in patterns.items():
        for match in re.finditer(pattern, sentence):
            start, end = match.span()
            for token in doc:
                if token.idx >= start and token.idx + len(token) <= end:
                    if tags[token.i] == "O":
                        tags[token.i] = f"B-{ent_type}"
                    else:
                        tags[token.i] = f"I-{ent_type}"
    return tags

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # Labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(simple_ner)
    predictions = predictions[['id', 'tags']]

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
