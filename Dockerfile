FROM python:3.12-slim

RUN useradd -m -u 1000 user && mkdir /app && chown user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user creditscore ./creditscore
COPY --chown=user api ./api
COPY --chown=user data/raw/train.csv.gz ./data/raw/train.csv.gz
COPY --chown=user data/raw/test.csv.gz ./data/raw/test.csv.gz

# Train during the build so the model always matches the installed library versions
RUN python -m creditscore.train

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
