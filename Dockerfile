# Gatekeeper gateway. Build: docker build -t gatekeeper . ; run with a
# sandbox .env file (never bake credentials into the image):
#   docker run --env-file .env -p 8000:8000 gatekeeper
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
