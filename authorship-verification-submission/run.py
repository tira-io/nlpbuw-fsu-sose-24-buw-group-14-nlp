from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira_code = Client()
    daf = tira_code.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions_model = model.predict(df["text"])
    df["generated"] = predictions_model
    daf = df[["id", "generated"]]

    # Save the predictions
    output = get_output(str(Path(__file__).parent))
    daf.to_json(
        Path(output) / "predictions.jsonl", orient="records", lines=True
    )
