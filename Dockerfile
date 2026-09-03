FROM python:3.14-slim

WORKDIR /app

# Primeiro as dependências, para aproveitar cache do Docker
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código necessário para inferência
COPY src ./src
COPY models ./models

# Usuário sem privilégios de root
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
