import os
import uuid
from typing import List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # 384 for all-MiniLM-L6-v2

# ------------------------------------------------------------------------------------
# DB connection
# ------------------------------------------------------------------------------------
conn = psycopg2.connect(dsn=DATABASE_URL)

# ------------------------------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------------------------------
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> List[float]:
    """
    Compute a single embedding as a Python list[float].
    """
    emb = model.encode([text])[0]
    return emb.astype(float).tolist()


def to_pgvector(embedding: List[float]) -> str:
    """
    Convert Python list to pgvector literal: [0.1,0.2,...]
    """
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
        )
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


# ------------------------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------------------------
class DocumentIn(BaseModel):
    user_id: str
    content: str
    metadata: Optional[dict] = None


class BulkDocumentsIn(BaseModel):
    documents: List[DocumentIn]


class DocumentOut(BaseModel):
    id: str
    user_id: str
    content: str
    metadata: Optional[dict]
    created_at: Optional[str]


class QueryIn(BaseModel):
    user_id: str
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    document: DocumentOut
    score: float


class QueryResponse(BaseModel):
    results: List[QueryResult]


# ------------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------------
app = FastAPI(title="Memory API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut)
def add_document(doc: DocumentIn):
    embedding = embed_text(doc.content)
    emb_vec = to_pgvector(embedding)
    doc_id = str(uuid.uuid4())

    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO app.documents (id, user_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                RETURNING id, user_id, content, metadata, created_at;
                """,
                (
                    doc_id,
                    doc.user_id,
                    doc.content,
                    Json(doc.metadata) if doc.metadata else None,
                    emb_vec,
                ),
            )
            row = cur.fetchone()

    row["created_at"] = (
        row["created_at"].isoformat() if row.get("created_at") else None
    )
    return DocumentOut(**row)


@app.post("/documents/bulk", response_model=List[DocumentOut])
def add_documents_bulk(req: BulkDocumentsIn):
    if not req.documents:
        return []

    results = []

    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for doc in req.documents:
                embedding = embed_text(doc.content)
                emb_vec = to_pgvector(embedding)
                doc_id = str(uuid.uuid4())

                cur.execute(
                    """
                    INSERT INTO app.documents (id, user_id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    RETURNING id, user_id, content, metadata, created_at;
                    """,
                    (
                        doc_id,
                        doc.user_id,
                        doc.content,
                        Json(doc.metadata) if doc.metadata else None,
                        emb_vec,
                    ),
                )
                row = cur.fetchone()
                row["created_at"] = (
                    row["created_at"].isoformat() if row.get("created_at") else None
                )
                results.append(DocumentOut(**row))

    return results


@app.post("/query", response_model=QueryResponse)
def query_documents(q: QueryIn):
    if q.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    embedding = embed_text(q.query)
    emb_vec = to_pgvector(embedding)

    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                  id,
                  user_id,
                  content,
                  metadata,
                  created_at,
                  1 - (embedding <-> %s::vector) AS score
                FROM app.documents
                WHERE user_id = %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (emb_vec, q.user_id, emb_vec, q.top_k),
            )
            rows = cur.fetchall()

    results: List[QueryResult] = []
    for row in rows:
        row["created_at"] = (
            row["created_at"].isoformat() if row.get("created_at") else None
        )
        doc = DocumentOut(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            metadata=row["metadata"],
            created_at=row["created_at"],
        )
        results.append(QueryResult(document=doc, score=float(row["score"])))

    return QueryResponse(results=results)

