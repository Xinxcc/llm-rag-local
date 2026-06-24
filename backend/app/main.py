import os
import re
import hashlib
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from pypdf import PdfReader
import docx


# -----------------------------
# Config
# -----------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_kb")

TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

API_KEY = os.getenv("API_KEY", "")  # set via .env / environment; never commit a real key

KNOWLEDGE_DIR = "/app/knowledge_sources"


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Local RAG Agent API", version="1.0.0")
qdrant = QdrantClient(url=QDRANT_URL, timeout=60)


# -----------------------------
# Schemas
# -----------------------------
class IngestRequest(BaseModel):
    subdir: Optional[str] = None
    access_group: Optional[str] = "internal"
    recreate_collection: bool = False


class IngestResponse(BaseModel):
    files_processed: int
    chunks_upserted: int
    collection: str


class QueryRequest(BaseModel):
    query: str
    access_group: Optional[str] = "internal"
    top_k: Optional[int] = None


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


# -----------------------------
# Helpers
# -----------------------------
def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server API key not configured. Set API_KEY in the environment (.env).",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if i >= n:
            break
    return chunks


def read_pdf(path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    pages = []
    for idx, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = clean_text(t)
        if t:
            pages.append({"page": idx + 1, "text": t})
    return pages


def read_docx(path: str) -> str:
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return clean_text("\n".join(parts))


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def ollama_embed(texts: List[str]) -> List[List[float]]:
    vectors = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=120,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return vectors


def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def ensure_collection(vector_size: int, recreate: bool = False):
    exists = False
    try:
        _ = qdrant.get_collection(QDRANT_COLLECTION)
        exists = True
    except Exception:
        exists = False

    if exists and recreate:
        qdrant.delete_collection(QDRANT_COLLECTION)
        exists = False

    if not exists:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def list_files(base_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.startswith("~$"):
                continue
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in [".pdf", ".txt", ".md", ".docx"]:
                files.append(path)
    return sorted(files)


def upsert_chunks(chunks: List[Dict[str, Any]], access_group: str, recreate_collection: bool) -> int:
    if not chunks:
        return 0

    vectors = ollama_embed([c["text"] for c in chunks])
    vector_size = len(vectors[0])
    ensure_collection(vector_size=vector_size, recreate=recreate_collection)

    points = []
    for c, v in zip(chunks, vectors):
        payload = {
            "source": c["source"],
            "page": c.get("page"),
            "chunk_id": c["chunk_id"],
            "access_group": access_group,
            "text": c["text"],
        }
        points.append(PointStruct(id=c["point_id"], vector=v, payload=payload))

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def retrieve(query: str, access_group: str, top_k: int) -> List[Dict[str, Any]]:
    qvec = ollama_embed([query])[0]
    flt = Filter(
        must=[FieldCondition(key="access_group", match=MatchValue(value=access_group))]
    )
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )
    out = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "source": p.get("source", ""),
                "page": p.get("page"),
                "chunk_id": p.get("chunk_id", ""),
                "text": p.get("text", ""),
                "score": h.score,
            }
        )
    return out


def build_prompt(user_query: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        src = c["source"]
        page = c.get("page")
        tag = f"[{i}] {src}" + (f" (p.{page})" if page else "")
        ctx_lines.append(tag + "\n" + c["text"])

    ctx_block = "\n\n---\n\n".join(ctx_lines) if ctx_lines else "(no evidence retrieved)"
    return f"""
You are an internal knowledge assistant. Answer using ONLY the information in the "Evidence" section.
If the evidence is not sufficient to answer, clearly say "I cannot confirm this from the indexed material" and state what is missing.
Every answer must include citation numbers, e.g. ...(see [1][3]).

User question:
{user_query}

Evidence:
{ctx_block}

Please output:
1) A concise answer
2) A list of citation numbers (e.g. [1], [2])
""".strip()


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "collection": QDRANT_COLLECTION}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    base_dir = KNOWLEDGE_DIR
    if req.subdir:
        base_dir = os.path.join(KNOWLEDGE_DIR, req.subdir)
        if not os.path.isdir(base_dir):
            raise HTTPException(status_code=400, detail="subdir not found")

    files = list_files(base_dir)
    all_chunks = []

    for fpath in files:
        rel = os.path.relpath(fpath, KNOWLEDGE_DIR)
        ext = os.path.splitext(fpath)[1].lower()

        if ext == ".pdf":
            pages = read_pdf(fpath)
            for pg in pages:
                chunks = chunk_text(pg["text"], CHUNK_SIZE, CHUNK_OVERLAP)
                for idx, ch in enumerate(chunks):
                    chunk_id = sha1(f"{rel}|p{pg['page']}|{idx}|{ch[:50]}")
                    point_id = int(chunk_id[:15], 16)
                    all_chunks.append(
                        {"source": rel, "page": pg["page"], "text": ch, "chunk_id": chunk_id, "point_id": point_id}
                    )

        elif ext == ".docx":
            text = read_docx(fpath)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                chunk_id = sha1(f"{rel}|{idx}|{ch[:50]}")
                point_id = int(chunk_id[:15], 16)
                all_chunks.append(
                    {"source": rel, "page": None, "text": ch, "chunk_id": chunk_id, "point_id": point_id}
                )

        else:  # .txt/.md
            text = read_text_file(fpath)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                chunk_id = sha1(f"{rel}|{idx}|{ch[:50]}")
                point_id = int(chunk_id[:15], 16)
                all_chunks.append(
                    {"source": rel, "page": None, "text": ch, "chunk_id": chunk_id, "point_id": point_id}
                )

    upserted = upsert_chunks(all_chunks, access_group=req.access_group or "internal", recreate_collection=req.recreate_collection)

    return IngestResponse(files_processed=len(files), chunks_upserted=upserted, collection=QDRANT_COLLECTION)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    top_k = req.top_k or TOP_K
    contexts = retrieve(req.query, access_group=req.access_group or "internal", top_k=top_k)

    prompt = build_prompt(req.query, contexts)
    answer = ollama_generate(prompt)

    citations = [
        Citation(source=c["source"], page=c.get("page"), chunk_id=c["chunk_id"])
        for c in contexts
    ]

    return QueryResponse(answer=answer, citations=citations)
