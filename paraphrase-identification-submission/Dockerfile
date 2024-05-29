FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD levenshtein.py /code/levenshtein.py
ADD model.joblib /code/model.joblib

RUN python3 -m nltk.downloader punkt

ENTRYPOINT [ "python3", "/code/run.py" ]
