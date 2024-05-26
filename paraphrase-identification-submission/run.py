from pathlib import Path
from joblib import load
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Compute the Levenshtein distance
    df["distance"] = levenshtein_distance(df)

    # Load the model (best threshold)
    model_path = Path(__file__).parent / "model.joblib"
    best_threshold = load(model_path)

    # Predict the labels
    df["label"] = (df["distance"] <= best_threshold).astype(int)
    df = df.drop(columns=["distance", "sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
