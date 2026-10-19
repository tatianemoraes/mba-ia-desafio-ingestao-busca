"""Retrieval and answer generation for the PDF question-answering CLI."""

import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector
import truststore

truststore.inject_into_ssl()

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não encontrei dados suficientes no documento para responder."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não encontrei dados suficientes no documento para responder."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não encontrei dados suficientes no documento para responder."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não encontrei dados suficientes no documento para responder."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"""

FALLBACK_ANSWER = "Não encontrei dados suficientes no documento para responder."


def _configuration() -> tuple[str, str, str, str]:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    database_url = os.getenv("DATABASE_URL", "").strip()
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "supertech_kb").strip()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Configure OPENAI_API_KEY no arquivo .env.")
        return provider, api_key, database_url, collection
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Configure GOOGLE_API_KEY no arquivo .env.")
        return provider, api_key, database_url, collection
    raise ValueError("LLM_PROVIDER deve ser 'openai' ou 'gemini'.")


def get_embeddings():
    provider, api_key, _, _ = _configuration()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    return GoogleGenerativeAIEmbeddings(
        model=model, google_api_key=api_key, transport="rest"
    )


def get_vector_store(embeddings=None, collection_name: str | None = None) -> PGVector:
    provider, _, database_url, collection = _configuration()
    if not database_url:
        raise ValueError("Configure DATABASE_URL no arquivo .env.")
    if embeddings is None:
        embeddings = get_embeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name or collection,
        connection=database_url,
        use_jsonb=True,
    )


def _get_llm():
    provider, api_key, _, _ = _configuration()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key, temperature=0, transport="rest"
    )


def search_prompt(
    question: str | None = None,
    collection_name: str | None = None,
    result_count: int = 10,
) -> dict | RunnableLambda:
    """Build an invoke-compatible QA chain or answer one question directly."""
    if question is None:
        return RunnableLambda(
            lambda user_question: search_prompt(str(user_question))["answer"]
        )

    if not question.strip():
        return {"answer": FALLBACK_ANSWER, "sources": []}

    vector_store = get_vector_store(get_embeddings(), collection_name=collection_name)
    results = vector_store.similarity_search_with_score(
        question.strip(), k=max(1, min(result_count, 20))
    )
    if not results:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = "\n\n---\n\n".join(document.page_content for document, _ in results)
    prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question.strip())
    response = _get_llm().invoke(prompt).content
    if isinstance(response, str):
        answer = response.strip()
    else:
        answer = " ".join(
            part.get("text", "") for part in response if isinstance(part, dict)
        ).strip()

    sources = []
    for document, score in results:
        page = document.metadata.get("page")
        reference = f"página {page + 1}" if isinstance(page, int) else None
        if reference is None:
            continue
        item = {"reference": reference, "score": float(score)}
        if item not in sources:
            sources.append(item)

    return {"answer": answer or FALLBACK_ANSWER, "sources": sources}
