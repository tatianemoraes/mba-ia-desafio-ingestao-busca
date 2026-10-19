"""Load a PDF, split it into chunks, and store embeddings in pgvector."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_embeddings, get_vector_store


def ingest_pdf() -> None:
    load_dotenv()
    pdf_path = Path(os.getenv("PDF_PATH", "document.pdf"))
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent.parent / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError(f"Nenhuma página foi extraída do PDF: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"Nenhum texto foi extraído do PDF: {pdf_path}")

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    # Replace the collection so repeated runs do not duplicate document chunks.
    vector_store.delete_collection()
    vector_store = get_vector_store(embeddings)

    batch_size = int(os.getenv("INGEST_BATCH_SIZE", "10"))
    delay_seconds = float(os.getenv("INGEST_BATCH_DELAY_SECONDS", "2"))
    if batch_size < 1:
        raise ValueError("INGEST_BATCH_SIZE precisa ser pelo menos 1.")
    if delay_seconds < 0:
        raise ValueError("INGEST_BATCH_DELAY_SECONDS não pode ser negativo.")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_documents(batch)
        processed = min(start + batch_size, len(chunks))
        print(f"Chunks armazenados: {processed}/{len(chunks)}")
        if processed < len(chunks) and delay_seconds > 0:
            time.sleep(delay_seconds)
    print(f"Ingestão concluída: {len(chunks)} chunks de {pdf_path.name}.")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        raise SystemExit(f"Falha na ingestão: {exc}") from exc
