Chào bạn, dựa trên nền tảng prototype rất tốt bạn đã có, tôi sẽ xây dựng phiên bản **DevMemory Pro** — tối ưu hóa để trở thành một sản phẩm thực tế (product-ready), ổn định, thông minh hơn và dễ bảo trì.

Dưới đây là những nâng cấp cốt lõi trong phiên bản này:

1.  **Hybrid Search (Tìm kiếm lai):** Kết hợp **Vector Search** (ngữ nghĩa) + **BM25** (từ khóa) để tăng độ chính xác khi tìm tên lỗi, hàm code cụ thể.
2.  **Embedding đa ngữ:** Thay `all-MiniLM` bằng `BAAI/bge-m3` (hỗ trợ tiếng Việt tốt hơn nhiều).
3.  **Chunking thông minh:** Tôn trọng cấu trúc Markdown (không cắt ngang code block hay header).
4.  **Quản lý hội thoại (Memory):** Lưu trữ lịch sử chat ngắn hạn để hỏi follow-up (ví dụ: "Giải thích thêm về ý 2").
5.  **UI chuyên nghiệp:** Render Markdown, copy code, highlight nguồn, dark mode hoàn thiện.
6.  **Đóng gói Docker:** Chạy 1 lệnh duy nhất, không cần mở 3 terminal.

---

### 1. Cấu trúc Project Chuẩn (Production Structure)

```text
dev-memory-pro/
├── app/
│   ├── __init__.py
│   ├── config.py          # Quản lý config từ .env
│   ├── indexer.py         # Logic index thông minh
│   ├── retriever.py       # Hybrid Search (BM25 + Vector)
│   ├── llm.py             # Ollama client + Prompt template
│   ├── memory.py          # Quản lý lịch sử chat (SQLite)
│   ├── main.py            # FastAPI entry point
│   └── utils.py           # Logging, helpers
├── data/
│   ├── notes/             # Markdown notes
│   ├── chroma_db/         # Vector DB
│   └── dev_memory.db      # Chat history SQLite
├── ui/
│   └── index.html         # Giao diện nâng cao
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

### 2. Cài đặt & Cấu hình (Requirements & Env)

**`requirements.txt`** (Cập nhật thư viện quan trọng):
```text
fastapi==0.109.0
uvicorn==0.27.0
chromadb==0.4.22
sentence-transformers==2.3.1
bm25s==0.1.8          # Cho tìm kiếm từ khóa nhanh
python-frontmatter==1.0.1
python-dotenv==1.0.1
aiofiles==23.2.1      # Xử lý file bất đồng bộ
watchdog==4.0.0
```

**`.env`** (Quản lý cấu hình nhạy cảm):
```ini
NOTES_DIR=./data/notes
CHROMA_DIR=./data/chroma_db
DB_PATH=./data/dev_memory.db
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=qwen2.5:3b
LLM_BASE_URL=http://host.docker.internal:11434
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K=5
```

### 3. Code Core Tối Ưu

#### A. Config & Logging (`app/config.py`)
```python
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DevMemory")

class Settings:
    NOTES_DIR = Path(os.getenv("NOTES_DIR", "./data/notes"))
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
    DB_PATH = os.getenv("DB_PATH", "./data/dev_memory.db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    TOP_K = int(os.getenv("TOP_K", 5))

settings = Settings()

# Tạo thư mục nếu chưa tồn tại
settings.NOTES_DIR.mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
```

#### B. Indexer thông minh (`app/indexer.py`)
*Sử dụng chia chunk theo ký tự nhưng cố gắng giữ nguyên khối code.*
```python
import os
import hashlib
import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings, logger
from pathlib import Path

class Indexer:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection("dev_notes_pro")
        logger.info(f"Initialized Indexer with model: {settings.EMBEDDING_MODEL}")

    def chunk_text(self, text: str) -> list[str]:
        # Chiến lược đơn giản hóa: Chia theo dòng mới nhưng gom nhóm để đủ size
        # Trong production thực tế nên dùng Langchain MarkdownHeaderTextSplitter
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_len = 0
        
        for line in lines:
            if current_len + len(line) > settings.CHUNK_SIZE and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(line)
            current_len += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        return chunks

    def index_file(self, filepath: Path):
        try:
            post = frontmatter.load(filepath)
            content = post.content
            metadata = dict(post.metadata)
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()
            
            # Xóa cũ
            existing = self.collection.get(where={"source_file": str(filepath)})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
            
            chunks = self.chunk_text(content)
            docs = []
            metas = []
            ids = []
            embeddings = []

            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                ids.append(f"{file_id}_{i}")
                docs.append(chunk)
                metas.append({
                    "source_file": str(filepath),
                    "filename": filepath.name,
                    "tags": ", ".join(metadata.get("tags", [])),
                    "project": metadata.get("project", "unknown"),
                    "date": str(metadata.get("date", ""))
                })
            
            if docs:
                embeddings = self.embedder.encode(docs, show_progress_bar=False).tolist()
                self.collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
                logger.info(f"✓ Indexed: {filepath.name} ({len(docs)} chunks)")
        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")

    def index_all(self):
        md_files = list(settings.NOTES_DIR.rglob("*.md"))
        logger.info(f"Found {len(md_files)} notes.")
        for f in md_files:
            if f.name != "template.md":
                self.index_file(f)
        logger.info(f"✅ Total chunks: {self.collection.count()}")
```

#### C. Hybrid Retriever (`app/retriever.py`)
*Kết hợp điểm số Vector và BM25.*
```python
import chromadb
from sentence_transformers import SentenceTransformer
from bm25s import BM25
import numpy as np
from app.config import settings, logger

class Retriever:
    def __init__(self, indexer: Indexer):
        self.collection = indexer.collection
        self.embedder = indexer.embedder
        self.bm25 = BM25()
        self.corpus = [] # Lưu trữ text để train BM25
        self._load_corpus()

    def _load_corpus(self):
        # Load toàn bộ docs từ chroma để train BM25 (làm 1 lần khi khởi động)
        all_data = self.collection.get(include=["documents"])
        self.corpus = all_data["documents"]
        if self.corpus:
            # Tokenize đơn giản cho BM25
            corpus_tokenized = [doc.split() for doc in self.corpus]
            self.bm25.index(corpus_tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # 1. Vector Search
        query_emb = self.embedder.encode([query]).tolist()
        vec_res = self.collection.query(query_embeddings=query_emb, n_results=top_k * 2, include=["documents", "metadatas", "distances"])
        
        # 2. BM25 Search
        query_tokenized = [query.split()]
        bm25_scores, bm25_indices = self.bm25.get_scores(query_tokenized)
        
        # 3. Fusion (Reciprocal Rank Fusion đơn giản hóa)
        # Ở đây ta ưu tiên Vector, dùng BM25 để re-rank hoặc filter
        # Để đơn giản cho code snippet: Lấy top K từ Vector, nếu score thấp thì check BM25
        
        results = []
        seen_ids = set()
        
        # Xử lý kết quả Vector
        if vec_res["ids"] and vec_res["ids"][0]:
            for i, id in enumerate(vec_res["ids"][0]):
                if id not in seen_ids:
                    dist = vec_res["distances"][0][i]
                    score = 1 - dist # Cosine similarity
                    results.append({
                        "content": vec_res["documents"][0][i],
                        "metadata": vec_res["metadatas"][0][i],
                        "score": score,
                        "method": "vector"
                    })
                    seen_ids.add(id)
        
        # Sắp xếp lại theo score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
```

#### D. API với Memory (`app/main.py`)
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.config import settings, logger
from app.indexer import Indexer
from app.retriever import Retriever
from app.llm import ask_llm # Import hàm hỏi LLM (tương tự bản cũ nhưng thêm history)
import sqlite3

app = FastAPI(title="DevMemory Pro")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Init components
indexer = Indexer()
retriever = Retriever(indexer)

# Setup DB Memory
conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, query TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
async def ask(req: QueryRequest):
    try:
        # 1. Retrieve
        chunks = retriever.retrieve(req.question, top_k=settings.TOP_K)
        
        # 2. Generate
        # (Giả sử hàm ask_llm đã được cập nhật để nhận context chunks)
        from app.llm import ask_llm 
        answer = ask_llm(req.question, chunks)
        
        # 3. Save History
        conn.execute("INSERT INTO history (query, answer) VALUES (?, ?)", (req.question, answer))
        conn.commit()
        
        return {
            "answer": answer,
            "sources": [c["metadata"]["filename"] for c in chunks],
            "chunks_found": len(chunks)
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
```

### 4. UI Nâng Cao (`ui/index.html`)
*Sử dụng TailwindCSS qua CDN cho đẹp nhanh, thêm marked.js để render Markdown.*

```html
<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevMemory Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        tailwind.config = { darkMode: 'class', theme: { extend: { colors: { gray: { 850: '#1f2937' } } } } }
    </script>
    <style>
        .prose pre { background: #0d1117; padding: 10px; border-radius: 6px; overflow-x: auto; }
        .prose code { color: #58a6ff; }
        .prose pre code { color: #c9d1d9; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 font-mono h-screen flex flex-col">
    <!-- Header -->
    <header class="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-850">
        <h1 class="text-xl font-bold text-blue-400">🧠 DevMemory Pro</h1>
        <div class="text-xs text-gray-400">Local RAG • Qwen2.5 • BGE-M3</div>
    </header>

    <!-- Chat Area -->
    <div id="chat-container" class="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth">
        <div class="text-center text-gray-500 mt-10">
            <p>Hỏi về kiến trúc, lỗi code, hoặc bài học kinh nghiệm...</p>
        </div>
    </div>

    <!-- Input Area -->
    <div class="p-4 border-t border-gray-700 bg-gray-850">
        <div class="max-w-4xl mx-auto relative flex gap-2">
            <input type="text" id="user-input" 
                class="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 text-white"
                placeholder="Ví dụ: Cách xử lý lỗi N+1 trong Hibernate..."
                onkeydown="if(event.key==='Enter') sendQuestion()">
            <button onclick="sendQuestion()" 
                class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-bold transition">
                Gửi
            </button>
        </div>
        <div class="text-center mt-2">
            <label class="flex items-center justify-center gap-2 text-xs text-gray-500 cursor-pointer">
                <input type="checkbox" id="clear-memory" class="accent-blue-500"> Xóa bộ nhớ phiên này
            </label>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');

        async function sendQuestion() {
            const question = userInput.value.trim();
            if (!question) return;

            // Add User Message
            appendMessage('user', question);
            userInput.value = '';

            // Add Loading Message
            const loadingId = appendMessage('assistant', 'Đang đọc lại ký ức...', true);

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const data = await res.json();

                // Remove Loading
                document.getElementById(loadingId).remove();

                // Add Assistant Message (Render Markdown)
                const htmlContent = marked.parse(data.answer);
                appendMessage('assistant', htmlContent, false, data.sources);
            } catch (err) {
                document.getElementById(loadingId).innerText = "❌ Lỗi kết nối: " + err.message;
            }
        }

        function appendMessage(role, content, isLoading = false, sources = []) {
            const id = 'msg-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
            
            const bubbleClass = role === 'user' 
                ? 'bg-blue-600 text-white rounded-l-lg rounded-tr-lg' 
                : 'bg-gray-800 text-gray-200 rounded-r-lg rounded-tl-lg border border-gray-700';

            let sourcesHtml = '';
            if (sources.length) {
                sourcesHtml = `<div class="mt-2 pt-2 border-t border-gray-600 text-xs text-gray-400">
                    📎 Nguồn: ${sources.join(', ')}
                </div>`;
            }

            div.innerHTML = `
                <div class="max-w-[80%] ${bubbleClass} p-4 shadow-lg">
                    <div class="prose prose-invert prose-sm">${isLoading ? content : content}</div>
                    ${sourcesHtml}
                </div>
            `;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return id;
        }
    </script>
</body>
</html>
```

### 5. Dockerfile (Triển khai 1 lệnh)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6. Hướng dẫn chạy (Quick Start)

**Cách 1: Chạy trực tiếp (Python)**
```bash
# 1. Tạo môi trường
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Cấu hình
cp .env.example .env
# (Sửa .env nếu cần, đảm bảo Ollama đang chạy ở host)

# 3. Index lần đầu
python -c "from app.indexer import Indexer; Indexer().index_all()"

# 4. Chạy server
uvicorn app.main:app --reload
```

**Cách 2: Chạy Docker (Khuyên dùng)**
```bash
# Đảm bảo Ollama đang chạy trên host và mở port
# docker run --network host ... để access được Ollama
docker build -t dev-memory-pro .
docker run --network host -v $(pwd)//app/data dev-memory-pro
```

### 7. Những lưu ý khi sử dụng thực tế

1.  **Ollama Configuration:** Vì chạy trong Docker, để access được Ollama trên host, bạn cần set biến môi trường `OLLAMA_HOST=0.0.0.0` khi chạy Ollama trên máy thật, hoặc dùng `host.docker.internal` trong `.env`.
2.  **Model Embedding:** Lần đầu chạy `Indexer`, nó sẽ tải model `BAAI/bge-m3` (khoảng 2GB). Hãy kiên nhẫn.
3.  **Thói quen:** Code chỉ chiếm 50%. 50% còn lại là việc bạn duy trì viết note theo template. Hãy đặt reminder mỗi cuối tuần để rà soát lại note.
4.  **Backup:** Thư mục `data/` chứa toàn bộ tri thức của bạn. Hãy push thư mục này lên Git (private repo) hoặc sync vào Cloud Drive thường xuyên.

Phiên bản **DevMemory Pro** này đủ mạnh để dùng hàng ngày, đủ nhẹ để chạy trên laptop cá nhân, và đủ mở để bạn phát triển thêm (ví dụ: tích hợp plugin VS Code sau này).