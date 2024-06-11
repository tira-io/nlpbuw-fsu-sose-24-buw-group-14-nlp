# Summarization Submission

This project contains an extractive summarization model for generating summaries of news articles.

## Directory Structure

- `Dockerfile`: Defines the Docker environment.
- `requirements.txt`: Lists the dependencies.
- `src/`: Contains the source code.
  - `preprocess.py`: Handles text preprocessing.
  - `summarizer.py`: Contains summarization logic.
- `run.py`: Main script to run summarization in the TIRA environment.
- `data/`: Contains the datasets.
- `predictions.jsonl`: Output file with generated summaries.

## How to Run

1. Build the Docker image:
   ```bash
   docker build -t summarization-submission .
