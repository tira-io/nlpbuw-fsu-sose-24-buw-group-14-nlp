FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD cosine_similarity.py /code/cosine_similarity.py
ADD model.joblib /code/paraphrase-identification-submission/model.joblib

RUN python3 -m nltk.downloader punkt

ENTRYPOINT [ "python3", "/code/run.py" ]
