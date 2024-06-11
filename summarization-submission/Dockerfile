FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD src/ /code/src/
ADD requirements.txt /code/requirements.txt
ADD data/ /code/data/

WORKDIR /code

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "/code/run.py"]
