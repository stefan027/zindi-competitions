FROM python:3.11.9-slim

WORKDIR /app

# Dependencies
COPY ./torch-requirements.txt .
COPY ./serve-requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir  -r torch-requirements.txt
RUN pip install --no-cache-dir  -r serve-requirements.txt

# Trained model and definition with main script
COPY ./saved_model /app/saved_model
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]
