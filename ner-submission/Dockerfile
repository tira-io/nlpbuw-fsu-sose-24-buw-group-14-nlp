# Use a specific NLP base image if it has necessary pre-configurations and tools
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Set the working directory in the Docker image
WORKDIR /code

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libssl-dev \
    curl \
    gcc \
    g++ \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Add the script to the image
ADD run.py /code/run.py

# Command to run on container start
ENTRYPOINT ["python3", "/code/run.py"]
