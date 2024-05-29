from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from cosine_similarity import compute_cosine_similarity
from joblib import load
import pandas as pd

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Compute the cosine similarity
    df["cosine_similarity"] = compute_cosine_similarity(df)

    # Load the best threshold
    best_threshold = load(Path(__file__).parent / "model.joblib")

    # Make predictions
    df["label"] = (df["cosine_similarity"] >= best_threshold).astype(int)
    df = df.drop(columns=["cosine_similarity", "sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
