from tira.rest_api_client import Client
from cosine_similarity import compute_cosine_similarity
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import numpy as np
import joblib

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    text["cosine_similarity"] = compute_cosine_similarity(text)
    df = text.join(labels)

    best_mcc = -1
    best_threshold = None
    for threshold in sorted(df["cosine_similarity"].unique()):
        y_true = df["label"]
        y_pred = (df["cosine_similarity"] >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    print(f"Best threshold: {best_threshold}")
    print(f"Best MCC: {best_mcc}")

    # Save the best threshold
    joblib.dump(best_threshold, 'model.joblib')
