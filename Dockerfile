FROM python:3.11-slim

WORKDIR /app

COPY requirements/requirements_train.txt .
# it takes a lot of time to install torch
RUN pip install --no-cache-dir -r requirements_train.txt

COPY src/ ./src
CMD ["python", "-u", "src/train.py"]
