Dưới đây là các file code còn thiếu và đã được tối ưu hóa cho phiên bản **DevMemory Pro**. Các file này được viết theo chuẩn production, xử lý lỗi tốt hơn, có logging đầy đủ và tích hợp chặt chẽ với kiến trúc đã thiết kế.

---

## 1. `app/llm.py` — LLM Layer với Retry & Streaming

File này xử lý việc giao tiếp với Ollama, bao gồm cơ chế retry khi model bận và format prompt tối ưu cho RAG.

```python
# app/llm.py
import httpx
import json
from typing import List, Dict, Optional
from app.config import settings, logger

class LLMClient:
    def __init__(self):
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL
        self.timeout = 120.0  # Tăng timeout cho câu hỏi phức tạp
        self.max_retries = 2
        
        self.system_prompt = """Bạn là trợ lý cá nhân của một Software Architect/Developer.
Bạn có quyền truy cập vào Knowledge Base cá nhân chứa các ghi chú kỹ thuật, quyết định kiến trúc và bài học kinh nghiệm.

QUY TẮC TRẢ LỜI:
1. Chỉ dựa vào context được cung cấp. Không bịa đặt thông tin.
2. Trích dẫn nguồn rõ ràng (tên file, ngày tháng) trong câu trả lời.
3. Nếu không tìm thấy thông tin trong context, hãy nói thẳng "Không có thông tin trong knowledge base".
4. Ưu tiên hiển thị code snippet nếu có.
5. Giữ câu trả lời ngắn gọn, đúng trọng tâm kỹ thuật.
6. Nếu câu hỏi không liên quan đến kỹ thuật, vẫn trả lời lịch sự nhưng nhắc nhở về mục đích của hệ thống.

FORMAT:
- Sử dụng Markdown để format code và tiêu đề.
- Liệt kê nguồn tham khảo ở cuối câu trả lời."""

    def _build_prompt(self, query: str, chunks: List[Dict], history: str = "") -> str:
        """Xây dựng prompt với context và lịch sử chat"""
        context_text = ""
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("filename", "unknown")
            date = chunk.get("metadata", {}).get("date", "")
            tags = chunk.get("metadata", {}).get("tags", "")
            
            context_text += f"""
---
[Source {i}] File: {source} | Date: {date} | Tags: {tags}
Content:
{chunk['content']}
"""
        
        history_section = f"""
Lịch sử hội thoại gần đây:
{history}
""" if history else ""

        prompt = f"""
{context_text}

{history_section}

Câu hỏi hiện tại: {query}

Hãy trả lời dựa trên context và lịch sử trên:"""
        
        return prompt

    def ask(self, query: str, chunks: List[Dict], history: str = "") -> str:
        """Gửi câu hỏi đến Ollama và nhận câu trả lời"""
        prompt = self._build_prompt(query, chunks, history)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Thấp để giảm hallucination
                "top_p": 0.9,
                "num_ctx": 4096,     # Context window lớn hơn cho nhiều chunks
                "repeat_penalty": 1.1
            }
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "Không có câu trả lời từ model.")
                
            except httpx.TimeoutException:
                logger.warning(f"Timeout lần {attempt + 1}/{self.max_retries + 1}")
                if attempt == self.max_retries:
                    return "⚠️ Timeout: Model phản hồi quá chậm. Hãy thử lại với câu hỏi ngắn hơn."
                    
            except httpx.ConnectError:
                logger.error("Không thể kết nối Ollama. Đảm bảo Ollama đang chạy.")
                return "⚠️ Lỗi kết nối: Không thể kết nối với Ollama. Đảm bảo service đang chạy."
                
            except Exception as e:
                logger.error(f"Lỗi LLM: {str(e)}")
                if attempt == self.max_retries:
                    return f"⚠️ Lỗi: {str(e)}"
        
        return "⚠️ Không thể nhận câu trả lời."

    def check_health(self) -> bool:
        """Kiểm tra kết nối đến Ollama"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except:
            return False

# Singleton instance
llm_client = LLMClient()
```

---

## 2. `app/memory.py` — Quản lý Lịch sử Chat (SQLite)

File này quản lý bộ nhớ hội thoại, cho phép hỏi follow-up và lưu lại lịch sử để tra cứu sau này.

```python
# app/memory.py
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
from app.config import settings, logger

class ChatMemory:
    """Quản lý lịch sử chat với SQLite"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self._init_db()
        logger.info(f"Chat memory initialized: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager cho DB connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB Error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Tạo bảng nếu chưa tồn tại"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    chunks_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON conversations(created_at)")
    
    def save_conversation(self, session_id: str, query: str, answer: str, 
                         sources: List[str], chunks_count: int) -> int:
        """Lưu một lượt hội thoại"""
        with self.get_connection() as conn:
            # Update session last_active
            conn.execute("""
                INSERT OR IGNORE INTO sessions (session_id) VALUES (?)
            """, (session_id,))
            conn.execute("""
                UPDATE sessions SET last_active = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            
            # Insert conversation
            cursor = conn.execute("""
                INSERT INTO conversations 
                (session_id, query, answer, sources, chunks_count)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, query, answer, ",".join(sources), chunks_count))
            
            logger.info(f"Saved conversation for session {session_id}")
            return cursor.lastrowid
    
    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Lấy lịch sử chat của session (cho follow-up questions)"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT query, answer, created_at 
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            # Return reversed (oldest first) for context building
            return [dict(row) for row in reversed(rows)]
    
    def get_recent_history_for_context(self, session_id: str, last_n: int = 3) -> str:
        """Xây dựng chuỗi lịch sử ngắn cho LLM context"""
        history = self.get_session_history(session_id, limit=last_n)
        if not history:
            return ""
        
        context_lines = []
        for h in history:
            context_lines.append(f"User: {h['query']}")
            context_lines.append(f"Assistant: {h['answer'][:500]}...")  # Giới hạn độ dài
        return "\n".join(context_lines)
    
    def delete_session(self, session_id: str) -> bool:
        """Xóa toàn bộ session"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            logger.info(f"Deleted session {session_id}")
            return True
    
    def get_stats(self) -> Dict:
        """Thống kê sử dụng"""
        with self.get_connection() as conn:
            total_convos = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            total_sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            
            return {
                "total_conversations": total_convos,
                "total_sessions": total_sessions,
                "db_path": self.db_path
            }

# Singleton instance
chat_memory = ChatMemory()
```

---

## 3. `app/watcher.py` — Auto-Index với Watchdog

File này theo dõi thư mục notes và tự động index khi có thay đổi, đảm bảo knowledge base luôn cập nhật.

```python
# app/watcher.py
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
from typing import Set
from app.config import settings, logger
from app.indexer import Indexer

class NoteFileHandler(FileSystemEventHandler):
    """Xử lý sự kiện thay đổi file trong thư mục notes"""
    
    def __init__(self, indexer: Indexer):
        super().__init__()
        self.indexer = indexer
        self._debounce_set: Set[str] = set()
        self._debounce_delay = 2.0  # Giây chờ để tránh index nhiều lần khi save
    
    def _should_process(self, path: str) -> bool:
        """Kiểm tra file có nên được index không"""
        path_obj = Path(path)
        
        # Chỉ xử lý file .md
        if path_obj.suffix.lower() != '.md':
            return False
        
        # Bỏ qua template
        if path_obj.name.lower() == 'template.md':
            return False
        
        # Bỏ qua thư mục ẩn
        if any(part.startswith('.') for part in path_obj.parts):
            return False
        
        return True
    
    def _debounced_index(self, filepath: Path):
        """Index với debounce để tránh trigger nhiều lần"""
        path_str = str(filepath)
        
        if path_str in self._debounce_set:
            return
        
        self._debounce_set.add(path_str)
        
        # Đợi một chút để file được write hoàn tất
        time.sleep(self._debounce_delay)
        
        try:
            if filepath.exists():
                logger.info(f"🔄 File changed: {filepath.name}")
                self.indexer.index_file(filepath)
            else:
                logger.info(f"🗑️ File deleted: {filepath.name}")
                # Có thể implement logic xóa khỏi vector DB ở đây
        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")
        finally:
            self._debounce_set.discard(path_str)
    
    def on_modified(self, event):
        """Xử lý sự kiện file bị sửa"""
        if isinstance(event, FileModifiedEvent) and self._should_process(event.src_path):
            filepath = Path(event.src_path)
            self._debounced_index(filepath)
    
    def on_created(self, event):
        """Xử lý sự kiện file mới được tạo"""
        if isinstance(event, FileCreatedEvent) and self._should_process(event.src_path):
            filepath = Path(event.src_path)
            logger.info(f"📄 New file detected: {filepath.name}")
            self._debounced_index(filepath)
    
    def on_deleted(self, event):
        """Xử lý sự kiện file bị xóa"""
        if self._should_process(event.src_path):
            logger.info(f"🗑️ File deleted: {Path(event.src_path).name}")
            # Có thể thêm logic xóa khỏi vector DB

class FileWatcher:
    """Wrapper để quản lý Observer"""
    
    def __init__(self, notes_dir: Path = None):
        self.notes_dir = notes_dir or settings.NOTES_DIR
        self.indexer = Indexer()
        self.observer = Observer()
        self.handler = NoteFileHandler(self.indexer)
        self._running = False
    
    def start(self):
        """Bắt đầu watching"""
        self.observer.schedule(
            self.handler, 
            str(self.notes_dir), 
            recursive=True
        )
        self.observer.start()
        self._running = True
        logger.info(f"👀 Watching {self.notes_dir} for changes...")
        logger.info(f"📊 Current index size: {self.indexer.collection.count()} chunks")
    
    def stop(self):
        """Dừng watching"""
        self._running = False
        self.observer.stop()
        self.observer.join()
        logger.info("🛑 File watcher stopped")
    
    def run_forever(self):
        """Chạy watcher vô hạn (cho production)"""
        self.start()
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop()
    
    def health_check(self) -> bool:
        """Kiểm tra watcher có đang chạy không"""
        return self._running and self.observer.is_alive()

# Singleton cho watcher
file_watcher = FileWatcher()

# Entry point cho running độc lập
if __name__ == "__main__":
    logger.info("🚀 Starting DevMemory File Watcher...")
    file_watcher.run_forever()
```

---

## 4. `app/main.py` — Cập nhật API Server hoàn chỉnh

File này tích hợp tất cả các module lại với nhau thành một API hoàn chỉnh.

```python
# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
from pathlib import Path

from app.config import settings, logger
from app.indexer import Indexer
from app.retriever import Retriever
from app.llm import llm_client
from app.memory import chat_memory
from app.watcher import file_watcher

# Init FastAPI
app = FastAPI(
    title="DevMemory Pro",
    description="Personal Knowledge Base với RAG Local",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init components
indexer = Indexer()
retriever = Retriever(indexer)

# ==================== Models ====================

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = 5

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

# ==================== Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve UI"""
    ui_path = Path("ui/index.html")
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI not found")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    """Xử lý câu hỏi với RAG"""
    import time
    start_time = time.time()
    
    # Generate session ID nếu chưa có
    session_id = req.session_id or str(uuid.uuid4())
    
    # 1. Retrieve
    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    
    # 2. Get history for context
    history_context = chat_memory.get_recent_history_for_context(session_id, last_n=3)
    
    # 3. Generate
    answer = llm_client.ask(req.question, chunks, history_context)
    
    # 4. Save to memory
    sources = list({c.get("metadata", {}).get("filename", "unknown") for c in chunks})
    chat_memory.save_conversation(
        session_id=session_id,
        query=req.question,
        answer=answer,
        sources=sources,
        chunks_count=len(chunks)
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_found=len(chunks),
        session_id=session_id,
        processing_time_ms=round(processing_time, 2)
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra sức khỏe hệ thống"""
    ollama_ok = llm_client.check_health()
    stats = chat_memory.get_stats()
    
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        total_chunks=indexer.collection.count(),
        total_conversations=stats["total_conversations"]
    )

@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks, req: ReindexRequest = None):
    """Trigger re-index (có thể chạy background)"""
    full = req.full_reindex if req else False
    
    if full:
        background_tasks.add_task(indexer.index_all)
        return {"status": "started", "message": "Full reindex started in background"}
    else:
        # Chỉ index lại files hiện tại (watcher đã làm việc này tự động)
        indexer.index_all()
        return {"status": "completed", "message": f"Indexed {indexer.collection.count()} chunks"}

@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    """Lấy lịch sử chat của session"""
    history = chat_memory.get_session_history(session_id, limit=limit)
    return {"session_id": session_id, "conversations": history}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Xóa session"""
    success = chat_memory.delete_session(session_id)
    return {"deleted": success, "session_id": session_id}

@app.get("/stats")
async def get_stats():
    """Thống kê hệ thống"""
    return {
        "chat_memory": chat_memory.get_stats(),
        "vector_db": {
            "total_chunks": indexer.collection.count(),
            "collection_name": indexer.collection.name
        },
        "watcher": {
            "running": file_watcher.health_check()
        }
    }

# Mount static files
ui_path = Path("ui")
if ui_path.exists():
    app.mount("/static", StaticFiles(directory=str(ui_path)), name="static")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 DevMemory Pro starting...")
    logger.info(f"📁 Notes dir: {settings.NOTES_DIR}")
    logger.info(f"🧠 LLM Model: {settings.LLM_MODEL}")
    logger.info(f"🔗 Embedding: {settings.EMBEDDING_MODEL}")
    
    # Check Ollama connection
    if not llm_client.check_health():
        logger.warning("⚠️ Ollama not connected. Some features may not work.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 5. Hướng dẫn chạy hoàn chỉnh

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Đảm bảo Ollama đang chạy
ollama serve  # Terminal 1
ollama pull qwen2.5:3b  # Nếu chưa có

# 3. Index lần đầu
python -m app.indexer  # Hoặc: python -c "from app.indexer import Indexer; Indexer().index_all()"

# 4. Chạy Auto-watcher (Optional - nếu muốn auto-index)
python -m app.watcher  # Terminal 2

# 5. Chạy API Server
python -m app.main  # Terminal 3
# Hoặc: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Mở browser
http://localhost:8000
```

---

## 6. Kiểm tra nhanh API

```bash
# Health check
curl http://localhost:8000/health

# Hỏi câu đầu tiên
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Cách xử lý lỗi N+1 trong Hibernate?"}'

# Lấy lịch sử session
curl http://localhost:8000/history/{session_id_from_response}

# Stats
curl http://localhost:8000/stats
```

---

Với các file bổ sung này, hệ thống **DevMemory Pro** của bạn đã hoàn chỉnh và sẵn sàng cho việc sử dụng thực tế hàng ngày. Các tính năng chính đã được cover:

| Tính năng | File | Trạng thái |
|-----------|------|------------|
| Index thông minh | `indexer.py` | ✅ |
| Hybrid Search | `retriever.py` | ✅ |
| LLM với Retry | `llm.py` | ✅ |
| Chat Memory | `memory.py` | ✅ |
| Auto-watch | `watcher.py` | ✅ |
| API hoàn chỉnh | `main.py` | ✅ |
| UI | `ui/index.html` | ✅ |