from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
import pandas as pd
from summarization import generate_summary

def main():
    try:
        # Initialize TIRA client and load data
        tira = Client()
        df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training")

        # Generate summaries for each story in the dataset
        df["summary"] = df["story"].apply(generate_summary)

        # Remove the original 'story' column
        df = df.drop(columns=["story"])

        # Save the predictions to a JSONL file
        output_directory = get_output_directory(str(Path(__file__).parent))
        output_path = Path(output_directory) / "predictions.jsonl"
        df.to_json(output_path, orient="records", lines=True)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
