# Use a specific NLP base image if it has necessary pre-configurations and tools
FROM fschlatt/natural-language-processing-exercises:0.0.1


WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libssl-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

ADD run.py /code/run.py

ENTRYPOINT ["python3", "/code/run.py"]
