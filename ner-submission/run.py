import spacy
import pandas as pd
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_model(model_path):
    """ Load the trained spaCy NER model from a directory. """
    return spacy.load(model_path)

def process_texts(nlp, texts):
    """ Process a list of texts and return the NER tags using BIO format. """
    results = []
    for text in texts:
        doc = nlp(text)
        tags = []
        for token in doc:
            if token.ent_iob_ == 'O':
                tags.append('O')
            else:
                tags.append(f"{token.ent_iob_}-{token.ent_type_}")
        results.append(tags)
    return results

if __name__ == "__main__":
    tira = Client()

    # Fetch input data using TIRA API
    text_data = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    sentences = text_data['sentence'].tolist()

    # Load the spaCy model
    model_path = "/workspaces/nlpbuw-fsu-sose-24-buw-group-14-nlp/ner-submission/ner_model/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(model_path)

    # Process the data
    predictions_tags = process_texts(nlp, sentences)

    # Prepare DataFrame for output  
    predictions_df = pd.DataFrame({
        "id": text_data['id'],
        "tags": predictions_tags
    })

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
