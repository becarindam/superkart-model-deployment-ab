FROM python:3.13-slim

WORKDIR /app

# Install the dependencies before copying the application files.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Include the API and trained pipeline in the container.
COPY app.py .
COPY superkart_model.joblib .

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
