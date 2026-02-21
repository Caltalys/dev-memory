# 🧠 DevMemory Pro

Personal Knowledge Base với RAG cục bộ — hỏi về kiến trúc, lỗi code, bài học kinh nghiệm từ Markdown notes cá nhân. Không cần cloud, chạy hoàn toàn trên máy local.

## Tính năng

- **Hybrid Search** — Vector (ChromaDB + BGE-M3) + BM25 với Reciprocal Rank Fusion
- **Date-Aware Retrieval** — Tự động lọc theo ngày khi hỏi "hôm nay", "2 tháng trước", `20/10/2025`, v.v.
- **Embedding đa ngữ** — `BAAI/bge-m3` hỗ trợ tiếng Việt tốt
- **Smart Chunking** — Chia chunk tôn trọng cấu trúc Markdown và code block
- **Chat Memory** — SQLite lưu lịch sử, hỏi follow-up được
- **Auto Re-index** — Watchdog tự động index khi note thay đổi
- **UI Dark Mode** — Markdown render, nguồn trích dẫn, health indicator (font Space Grotesk)
- **Docker Ready** — Chạy 1 lệnh
- **Agent Workflow** — `/create-dev-note` để tạo note kinh nghiệm theo template tự động

## Quick Start

### Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **Python** | 3.10+ |
| **RAM** | 4GB+ (khuyên dùng 8GB+) |
| **Disk** | 5GB+ (model embedding ~2GB + model LLM ~2GB) |
| **Internet** | Cần khi tải model lần đầu |
| **Ollama** | Bất kỳ version nào |

### Cài đặt Python (Ubuntu/Debian)

> Bỏ qua nếu đã có Python 3.10+ (`python3 --version`).

```bash
# Cài Python và công cụ venv
sudo apt update
sudo apt install python3 python3.12-venv python3-pip -y

# Kiểm tra phiên bản
python3 --version  # Python 3.12.x
```

> **Lưu ý:** Trên Ubuntu/Debian, luôn dùng `python3` thay vì `python`.

### Cài đặt Ollama và chọn LLM

```bash
# Cài Ollama (nếu chưa có)
curl -fsSL https://ollama.com/install.sh | sh

# Khởi động Ollama service
ollama serve &

# Chọn model LLM phù hợp với RAM máy:
ollama pull qwen2.5:1.5b  # 🟢 RAM thấp (~1.5GB) — Khuyên dùng cho máy <8GB RAM
ollama pull qwen2.5:3b    # 🟡 Cân bằng (~2.5GB) — Cần 8GB+ RAM
ollama pull qwen2.5:7b    # 🔴 Chất lượng cao (~5GB) — Cần GPU hoặc 16GB+ RAM
```

### Cách 1: Python Local (Khuyên dùng khi dev)

```bash
# 1. Clone hoặc cd vào thư mục project
cd /path/to/dev-memory

# 2. Tạo và kích hoạt virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate.bat       # Windows

# 3. Cài dependencies (CPU-only — không cần GPU/CUDA)
pip install -r requirements.txt
# ⏳ Lần đầu mất 3-5 phút (tải PyTorch CPU, ChromaDB...)

# 4. Cấu hình môi trường
cp .env.example .env
```

Sửa file `.env` nếu cần — đặc biệt 2 dòng này:

```ini
# Đổi model LLM nếu cần (xem phần Chọn LLM ở trên)
LLM_MODEL=qwen2.5:1.5b

# Ollama URL: localhost nếu chạy native, host.docker.internal nếu trong Docker
LLM_BASE_URL=http://localhost:11434
```

```bash
# 5. Đặt notes vào data/notes/ (xem template)
# Đã có sẵn: data/notes/template.md

# 6. Chạy server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# ⏳ Lần đầu: tải model BAAI/bge-m3 (~2GB) — mất 5-15 phút
# ✅ Từ lần 2 trở đi: khởi động trong ~30 giây
```

Mở trình duyệt: **http://localhost:8000**

### Cách 2: Docker

```bash
# 1. Build và chạy
docker compose up -d

# Xem log
docker compose logs -f
```

> **Lưu ý:** Đảm bảo Ollama đang chạy và `LLM_BASE_URL` trong `.env` trỏ đúng đến Ollama.

## Cấu trúc Project

```
dev-memory/
├── app/
│   ├── config.py       # Settings từ .env
│   ├── indexer.py      # Smart chunking + ChromaDB indexing
│   ├── retriever.py    # Hybrid Search + Date-Aware Filter
│   ├── llm.py          # Ollama client + Retry + Optimized Prompt
│   ├── memory.py       # Chat history SQLite
│   ├── watcher.py      # Auto re-index khi note thay đổi
│   └── main.py         # FastAPI server
├── data/
│   ├── notes/          # 📝 ĐẶT MARKDOWN NOTES VÀO ĐÂY
│   │   └── template.md
│   ├── chroma_db/      # Vector index (auto-generated)
│   └── dev_memory.db   # Chat history (auto-generated)
├── ui/
│   └── index.html      # Web UI (Space Grotesk)
├── .agents/
│   ├── rules.md        # Quy tắc AI agent cho project
│   └── workflows/
│       └── create-dev-note.md  # Workflow tạo note kinh nghiệm
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## API Endpoints

| Endpoint | Method | Mô tả |
|---|---|---|
| `/` | GET | Web UI |
| `/ask` | POST | Hỏi RAG |
| `/ask/stream` | POST | Hỏi RAG — Streaming |
| `/health` | GET | Trạng thái hệ thống |
| `/reindex` | POST | Trigger re-index thủ công |
| `/history/{session_id}` | GET | Lịch sử chat |
| `/session/{session_id}` | DELETE | Xóa session |
| `/stats` | GET | Thống kê |

### Ví dụ gọi API

```bash
# Hỏi câu hỏi
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Cách xử lý lỗi N+1 trong SQLAlchemy?"}'

# Health check
curl http://localhost:8000/health

# Re-index tất cả notes
curl -X POST http://localhost:8000/reindex \
  -H "Content-Type: application/json" \
  -d '{"full_reindex": true}'
```

## Format Note (.md)

Dùng template `data/notes/template.md` để tạo note chuẩn:

```markdown
---
tags: [python, fastapi]
project: my-project
date: 2026-02-21
---

# Tiêu đề Note
...
```

## Lưu ý Quan Trọng

1. **Lần đầu chạy** — Model `BAAI/bge-m3` (~2GB) sẽ được tải tự động. Cần Internet và kiên nhẫn.
2. **Ollama trong Docker** — Nếu Ollama chạy trên host, đổi `LLM_BASE_URL=http://host.docker.internal:11434` trong `.env`.
3. **Backup data** — Thư mục `data/notes/` là toàn bộ tri thức của bạn. Push lên Git private repo hoặc sync lên Cloud Drive thường xuyên.
4. **Thói quen** — Tool chỉ hiệu quả khi bạn duy trì viết note. Đặt reminder cuối tuần để review.

## Khắc Phục Sự Cố

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `python not found` | Ubuntu dùng `python3` | Dùng `python3` thay vì `python` |
| `ensurepip not available` | Thiếu gói venv | `sudo apt install python3.12-venv` |
| `externally-managed-environment` | Không dùng venv | Tạo venv trước: `python3 -m venv venv` |
| `np.float_ removed` | NumPy 2.x không tương thích | `pip install "numpy>=1.24.0,<2.0"` |
| `bm25s==0.1.8 not found` | Version bị yanked | Đã fix trong `requirements.txt` (dùng `0.1.10`) |
| `AssertionError` khi `pip install` | pip version cũ | `pip install --upgrade pip` |
| `No space left on device` | CUDA torch (~2GB) tải đầy disk | `requirements.txt` đã dùng CPU-only torch |
| `Cannot connect to Ollama` | Ollama chưa chạy | `ollama serve` hoặc kiểm tra port 11434 |
| Phản hồi chậm (>2 phút) | Model quá lớn cho CPU | Đổi sang `qwen2.5:1.5b` trong `.env` |
| OOM / Server crash | RAM không đủ | Đổi model nhỏ hơn hoặc tắt bớt app |
| Câu trả lời lẫn lộn | Model nhỏ, context nhiễu | Hỏi cụ thể hơn, thêm từ khóa thời gian |
