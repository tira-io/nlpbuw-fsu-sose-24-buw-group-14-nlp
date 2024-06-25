FROM python:3.9-slim
FROM fschlatt/natural-language-processing-exercises:0.0.1

WORKDIR /code

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*


RUN pip install tira rest_api_client pandas sklearn_crfsuite matplotlib seqeval evaluate

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /code

CMD ["python3", "run.py"]
