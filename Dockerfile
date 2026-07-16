FROM python:3.13-slim

# Must be set before the warm step so build-time download and runtime load
# resolve to the same cache path.
ENV HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU torch first, from PyTorch's own index. Installing it up front means pip
# sees torch already satisfied and won't pull the CUDA build as a transitive
# dependency of sentence-transformers.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Warm BGE-M3 via the exact class src/rag.py uses.
RUN python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='BAAI/bge-m3')"

# Code last: edits here won't invalidate the 2.3 GB model layer above.
COPY app.py .
COPY src/ src/
COPY eval/ eval/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /opt/hf-cache /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
