FROM fschlatt/natural-language-processing-exercises:0.0.1

# Set the working directory
WORKDIR /code

# Add the source code and requirements
ADD src/ /code/src/
ADD requirements.txt /code/requirements.txt
ADD run.py /code/run.py

# Install the dependencies
RUN pip install -r requirements.txt

# Define the command to run the application
CMD ["python", "run.py"]
