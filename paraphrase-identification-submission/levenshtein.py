import nltk
import pandas as pd

def levenshtein_distance(df: pd.DataFrame):
    text = df[['sentence1', 'sentence2']].apply(lambda row: (nltk.word_tokenize(row['sentence1']), nltk.word_tokenize(row['sentence2'])), axis=1)
    distance = text.apply(lambda x: nltk.edit_distance(x[0], x[1]))
    return distance
