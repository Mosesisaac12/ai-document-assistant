from io import BytesIO
import hashlib
import os
import chromadb
import ollama
from pypdf import PdfReader


EMBEDDING_MODEL = "nomic-embed-text"
REWRITE_MODEL = "llama3.2"
ANSWER_MODEL = "granite4.2:8b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 5


def get_chroma_client():
    chroma_host = os.getenv("CHROMA_HOST")
    chroma_port= int(os.getenv("CHROMA_PORT","8000"))

    return chromadb.HttpClient(
            host=chroma_host,
            port = chroma_port
        )




def create_embedding(text):
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]


def create_chat_response(messages):
    response = ollama.chat(
        model=ANSWER_MODEL,
        messages=messages,
        think=False,
        keep_alive="10m",
        options={
            "temperature": 0.0,
            "num_predict": 300
        }
    )

    return response["message"]["content"]


def extract_pdf_chunks(pdf_bytes):
    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    page_chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text() or ""

        for start_position in range(
            0,
            len(page_text),
            CHUNK_SIZE - CHUNK_OVERLAP
        ):
            chunk_text = page_text[
                start_position:
                start_position + CHUNK_SIZE
            ]

            if chunk_text.strip():
                page_chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

    if not page_chunks:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be scanned or image-based."
        )

    return reader, page_chunks


def get_document_identity(pdf_bytes):
    indexing_configuration = (
        f"{CHUNK_SIZE}|"
        f"{CHUNK_OVERLAP}|"
        f"{EMBEDDING_MODEL}"
    ).encode("utf-8")

    identity_data = (
        pdf_bytes
        + indexing_configuration
    )

    pdf_id = hashlib.sha256(
        identity_data
    ).hexdigest()

    collection_name = f"pdf_{pdf_id[:16]}"

    return pdf_id, collection_name


def index_document(
    client,
    pdf_bytes,
    source_name
):
    reader, page_chunks = extract_pdf_chunks(
        pdf_bytes
    )

    pdf_id, collection_name = (
        get_document_identity(pdf_bytes)
    )

    collection = client.get_or_create_collection(
        name=collection_name
    )

    was_already_indexed = collection.count() > 0

    if not was_already_indexed:
        chunk_ids = []
        chunk_documents = []
        chunk_metadatas = []
        chunk_embeddings = []

        for chunk_number, chunk in enumerate(
            page_chunks
        ):
            chunk_text = chunk["text"]
            page_number = chunk["page"]

            chunk_embedding = create_embedding(
                chunk_text
            )

            chunk_ids.append(
                f"{pdf_id}_chunk_{chunk_number}"
            )

            chunk_documents.append(
                chunk_text
            )

            chunk_metadatas.append(
                {
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "source": source_name
                }
            )

            chunk_embeddings.append(
                chunk_embedding
            )

        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings
        )

    return {
        "collection": collection,
        "pdf_id": pdf_id,
        "collection_name": collection_name,
        "page_count": len(reader.pages),
        "chunk_count": collection.count(),
        "already_indexed": was_already_indexed
    }


def create_conversation_history(
    messages,
    maximum_messages=6
):
    recent_messages = messages[
        -maximum_messages:
    ]

    return "\n".join(
        (
            f"{message['role']}: "
            f"{message['content']}"
        )
        for message in recent_messages
    )


def rewrite_question(
    user_question,
    conversation_history
):
    if not conversation_history.strip():
        return user_question

    try:
        response = ollama.chat(
            model=REWRITE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest question as a "
                        "standalone question using the conversation "
                        "history. Resolve references such as he, she, "
                        "they, it, this, that, its, and their. Do not "
                        "answer the question or add new facts. If the "
                        "question is already standalone, return it "
                        "unchanged. Return only the rewritten question."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONVERSATION HISTORY:\n"
                        f"{conversation_history}\n\n"
                        f"LATEST QUESTION:\n"
                        f"{user_question}"
                    )
                }
            ],
            think=False,
            keep_alive="10m",
            options={
                "temperature": 0
            }
        )

        rewritten_question = (
            response["message"]["content"].strip()
        )

        if rewritten_question:
            return rewritten_question

    except Exception:
        pass

    return user_question


def retrieve_documents(
    collection,
    question
):
    question_embedding = create_embedding(
        question
    )

    result_count = min(
        RETRIEVAL_COUNT,
        collection.count()
    )

    search_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return {
        "documents": search_results["documents"][0],
        "metadatas": search_results["metadatas"][0],
        "distances": search_results["distances"][0]
    }


def build_context(
    documents,
    metadatas
):
    context_parts = []
    source_pages = set()

    for document, metadata in zip(
        documents,
        metadatas
    ):
        page_number = metadata["page"]

        context_parts.append(
            f"[Page {page_number}]\n{document}"
        )

        source_pages.add(page_number)

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, sorted(source_pages)


def build_answer_prompt(
    context,
    conversation_history,
    original_question,
    standalone_question,
):
    return f"""
You are an AI document assistant.

Answer the question using only the supplied document context.

Rules:

1. Give a direct, concise answer to the question.

2. Every factual claim must be explicitly supported by the context.

3. Preserve exact relationships between names, categories, model
   variants, numerical values, dates, and results.

4. Do not guess a missing relationship or determine it through
   process of elimination.

5. When reading a table, carefully associate every value with its
   correct row and column headings.

6. If two passages report conflicting values, report both values and
   clearly state that the document contains a discrepancy. Do not
   silently choose one value.

7. Cite supporting pages using [Page X].

8. If the context contains no relevant information, say:
   "The answer is not available in the uploaded document."

9. Do not mention retrieval, chunks, embeddings, prompts, vector
   databases, or internal processing.

DOCUMENT CONTEXT:

{context}

QUESTION:

{standalone_question}
"""

def generate_answer(model_prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded document assistant. "
                "Use language understanding and reasoning to interpret "
                "questions, but treat the supplied document context as "
                "the authoritative source for factual claims."
            )
        },
        {
            "role": "user",
            "content": model_prompt
        }
    ]

    return create_chat_response(messages)


def answer_question(
    collection,
    user_question,
    previous_messages
):
    conversation_history = (
        create_conversation_history(
            previous_messages
        )
    )

    standalone_question = rewrite_question(
        user_question,
        conversation_history
    )

    retrieval_results = retrieve_documents(
        collection,
        standalone_question
    )

    context, source_pages = build_context(
        retrieval_results["documents"],
        retrieval_results["metadatas"]
    )

    model_prompt = build_answer_prompt(
        context=context,
        conversation_history=conversation_history,
        original_question=user_question,
        standalone_question=standalone_question
    )

    answer = generate_answer(
        model_prompt
    )

    return {
        "answer": answer,
        "standalone_question": standalone_question,
        "context": context,
        "source_pages": source_pages,
        "documents": retrieval_results["documents"],
        "metadatas": retrieval_results["metadatas"],
        "distances": retrieval_results["distances"]
    }
