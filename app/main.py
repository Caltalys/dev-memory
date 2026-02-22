import time
import uuid
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings, logger
from app.indexer import Indexer
from app.retriever import Retriever
from app.llm import llm_client
from app.memory import chat_memory
from app.watcher import FileWatcher

# ── Init ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="DevMemory Pro", description="Personal Knowledge Base RAG", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

indexer = Indexer()
retriever = Retriever(indexer)
file_watcher = FileWatcher(indexer=indexer)

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = 3  # Giảm từ 5→3: ít chunk hơn = prompt ngắn hơn = nhanh hơn

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_found: int
    session_id: str
    processing_time_ms: float

class ReindexRequest(BaseModel):
    full_reindex: bool = False

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    total_chunks: int
    total_conversations: int

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = Path("ui/index.html")
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI not found. Did you create ui/index.html?")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    """Xử lý câu hỏi RAG với lịch sử hội thoại."""
    start = time.time()
    session_id = req.session_id or str(uuid.uuid4())

    # 1. Retrieve
    chunks = retriever.retrieve(req.question, top_k=req.top_k)

    # 2. History context
    history_context = chat_memory.get_recent_history_for_context(session_id, last_n=2)

    # 3. Generate (non-streaming, blocks until done)
    answer = llm_client.ask(req.question, chunks, history_context)

    # 4. Save to memory
    sources = list({c.get("metadata", {}).get("filename", "unknown") for c in chunks})
    chat_memory.save_conversation(
        session_id=session_id,
        query=req.question,
        answer=answer,
        sources=sources,
        chunks_count=len(chunks),
    )

    processing_ms = round((time.time() - start) * 1000, 2)
    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_found=len(chunks),
        session_id=session_id,
        processing_time_ms=processing_ms,
    )


@app.post("/ask/stream")
async def ask_stream(req: QueryRequest):
    """Streaming endpoint — trả về tokens ngay lập tức qua SSE."""
    session_id = req.session_id or str(uuid.uuid4())
    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    history_context = chat_memory.get_recent_history_for_context(session_id, last_n=2)
    sources = list({c.get("metadata", {}).get("filename", "unknown") for c in chunks})

    # Gửi metadata trước (sources, session_id)
    meta_event = json.dumps({"type": "meta", "session_id": session_id, "sources": sources, "chunks_found": len(chunks)})

    full_answer_parts = []

    def event_stream():
        # 1. Gửi metadata ngay
        yield f"data: {meta_event}\n\n"

        # 2. Stream từng token từ LLM
        for token in llm_client.ask_stream(req.question, chunks, history_context):
            full_answer_parts.append(token)
            token_event = json.dumps({"type": "token", "text": token})
            yield f"data: {token_event}\n\n"

        # 3. Done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # 4. Save to memory sau khi stream xong
        full_answer = "".join(full_answer_parts)
        try:
            chat_memory.save_conversation(
                session_id=session_id,
                query=req.question,
                answer=full_answer,
                sources=sources,
                chunks_count=len(chunks),
            )
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_ok = llm_client.check_health()
    stats = chat_memory.get_stats()
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        total_chunks=indexer.collection.count(),
        total_conversations=stats["total_conversations"],
    )


@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks, req: ReindexRequest = ReindexRequest()):
    if req.full_reindex:
        background_tasks.add_task(_do_reindex)
        return {"status": "started", "message": "Full reindex started in background"}
    indexer.index_all()
    retriever.reload_corpus()
    return {"status": "completed", "chunks": indexer.collection.count()}


def _do_reindex():
    indexer.index_all()
    retriever.reload_corpus()


@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    history = chat_memory.get_session_history(session_id, limit=limit)
    return {"session_id": session_id, "conversations": history}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    success = chat_memory.delete_session(session_id)
    return {"deleted": success, "session_id": session_id}


@app.get("/stats")
async def get_stats():
    return {
        "chat_memory": chat_memory.get_stats(),
        "vector_db": {
            "total_chunks": indexer.collection.count(),
            "collection": indexer.collection.name,
        },
        "watcher": {"running": file_watcher.health_check()},
    }

# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 DevMemory Pro starting...")
    logger.info(f"📁 Notes: {settings.NOTES_DIR}")
    logger.info(f"🧠 LLM:   {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    logger.info(f"🔗 Model: {settings.EMBEDDING_MODEL}")

    # Index nếu chưa có gì
    if indexer.collection.count() == 0:
        logger.info("Empty index — running initial indexing...")
        indexer.index_all()
        retriever.reload_corpus()

    # Bắt đầu file watcher
    file_watcher.start()

    if not llm_client.check_health():
        logger.warning("⚠️  Ollama not reachable. /ask will return error messages until Ollama starts.")

    # Warmup dateparser — load locale data trước để tránh delay ở request đầu tiên
    import asyncio
    asyncio.get_event_loop().run_in_executor(None, _warmup_dateparser)


def _warmup_dateparser():
    """Tải trước locale data của dateparser ở background để tránh cold-start."""
    try:
        import dateparser
        dateparser.parse("today")
        logger.info("✅ dateparser warmed up.")
    except Exception as e:
        logger.warning(f"dateparser warmup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    file_watcher.stop()
    logger.info("👋 DevMemory Pro stopped.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
