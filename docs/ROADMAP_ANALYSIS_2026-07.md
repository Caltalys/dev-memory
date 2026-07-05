# Phân Tích Cạnh Tranh & Roadmap Khả Thi — DevMemory Pro

**Ngày tạo**: 2026-07-05
**Phương pháp**: Mọi số liệu repo (stars, trạng thái hoạt động) được kiểm chứng trực tiếp trên GitHub ngày 2026-07-05. Mọi đánh giá độ khả thi đối chiếu với code thực tế trong `app/`.
**Thay thế**: `FEATURE_ANALYSIS_REPORT.md` và `DEEP_DIVE_ANALYSIS_5_FEATURES.md` (chứa số liệu chưa kiểm chứng và code mẫu không khớp codebase — xem banner trong 2 file đó).

---

## 1. DevMemory Pro hiện tại (baseline thực tế)

| Thành phần | Hiện trạng (đã đọc code) |
|---|---|
| Indexing | Semantic Sectioning theo heading `##`, phân loại section (`symptom`/`root_cause`/`solution`/...), tôn trọng code block — `app/indexer.py` |
| Retrieval | Hybrid: Vector (ChromaDB + BGE-M3) + BM25 (bm25s) + Reciprocal Rank Fusion; Date-aware filter qua `dateparser` — `app/retriever.py` |
| LLM | Ollama `/api/generate`, streaming, system prompt chống hallucination — `app/llm.py` |
| Note format | Markdown + YAML frontmatter theo hướng **OKF** (`type`, `tags`, `project`, `timestamp`) — từ PR #1 |
| Chat memory | SQLite — `app/memory.py` |
| UI | Vanilla HTML/JS, dark mode — `ui/index.html` |
| Ràng buộc thiết kế | **Single-user, local-first, CPU-only, tiếng Việt là ngôn ngữ chính** |

Điểm khác biệt thật sự so với thị trường: **date-aware retrieval + hybrid search + embedding đa ngữ tốt cho tiếng Việt + note format chuẩn mở (OKF)**, chạy nhẹ trên CPU.

---

## 2. Bức tranh thị trường (số liệu đã kiểm chứng 2026-07-05)

### 2.1 Nhóm "general RAG app" — KHÔNG cạnh tranh trực tiếp

| Repo | Stars | Trạng thái | Ghi chú |
|---|---|---|---|
| [anything-llm](https://github.com/Mintplex-Labs/anything-llm) | 62.6k | Rất active (v1.15.0, 06/2026) | All-in-one private RAG, multi-user, 30+ LLM providers |
| [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | 38.3k | Active | RAG + Agent offline, tối ưu tiếng Trung |
| [khoj](https://github.com/khoj-ai/khoj) | 35.5k | Active (2.0 beta, 03/2026) | "AI second brain", PDF/markdown/Notion, client Obsidian |

**Kết luận**: Segment "app RAG tổng quát" đã có người khổng lồ chiếm giữ. DevMemory **không nên** đua tính năng ở đây (multi-user, RBAC, đa nền tảng...). Giá trị của DevMemory là công cụ cá nhân nhẹ, chuyên cho dev notes tiếng Việt.

### 2.2 Nhóm "personal knowledge + AI" — cạnh tranh/tham khảo trực tiếp

| Repo | Stars | Trạng thái | Bài học cho DevMemory |
|---|---|---|---|
| [claude-obsidian](https://github.com/AgriciDaniel/claude-obsidian) | 8.8k | Active | Nhu cầu "AI tự tổ chức vault markdown" là có thật và lớn |
| [reor](https://github.com/reorproject/reor) | 8.6k | ⚠️ **Archived 03/2026** | App desktop (Electron) local AI notes có 8.6k⭐ vẫn chết — cảnh báo cho hướng "viết lại desktop app" |
| [obsidian-smart-connections](https://github.com/brianpetro/obsidian-smart-connections) | 5.2k | Active (06/2026) | "Related notes" bằng local embedding là feature được ưa chuộng nhất trong nhóm này |
| [basic-memory](https://github.com/basicmachines-co/basic-memory) | 3.4k | Active (v0.22.1, 06/2026) | **Anh em kiến trúc gần nhất**: markdown KB + MCP server (FastMCP), knowledge graph từ wikilinks. Chứng minh niche "MCP memory backend" có traction |

### 2.3 Hạ tầng khả dụng (để đánh giá effort)

| Repo | Stars | Ý nghĩa |
|---|---|---|
| [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | 23.5k | MCP server chỉ cần ~15 dòng code với SDK chính thức — **không cần tự viết JSON-RPC** |
| [microsoft/markitdown](https://github.com/microsoft/markitdown) | 163k | PDF/docx/xlsx/ảnh → Markdown bằng 3 dòng Python — **không cần tự viết PDF extraction** |

### 2.4 Số liệu sai trong báo cáo cũ (đã kiểm chứng lại)

| Claim cũ | Thực tế |
|---|---|
| Langchain-Chatchat "6.2K ⭐" | 38.3k ⭐ (sai ~6 lần) |
| imbuto-knowledge-os — dùng làm reference kiến trúc chính | 0 ⭐, 2 commits — repo gần như rỗng |
| mcp-notes — "Priority 1: Study first" | 1 ⭐ |

---

## 3. Đánh giá tính năng — đối chiếu codebase thật

### ✅ NÊN LÀM

#### F1. MCP Server (ưu tiên cao nhất)

- **Giá trị**: Dùng DevMemory làm "memory backend" trực tiếp từ Claude Code/Claude Desktop/Cursor. `basic-memory` (3.4k⭐) chứng minh nhu cầu này.
- **Cách làm thật**: Dùng MCP Python SDK chính thức (decorator `@mcp.tool()`), **tái sử dụng singleton có sẵn** — `Retriever.retrieve()` và `LLMClient.ask()`. Không tự viết JSON-RPC như báo cáo cũ đề xuất.
- **Tools đề xuất**: `search_notes(query, top_k)`, `ask(question)`, `list_notes(tag?)` — bắt đầu **read-only**; tool ghi note (`create_note`) để giai đoạn sau kèm validation frontmatter OKF, vì tool ghi mở ra rủi ro prompt-injection ghi bẩn vào knowledge base.
- **Effort**: ~2–4 ngày (150–300 LOC + entry `python -m app.mcp` + docs cấu hình client).

#### F2. Wikilink & OKF cross-link trong indexer

- **Giá trị**: Nền tảng cho backlinks, related notes, graph, và tương thích Obsidian. OKF spec vốn đã coi link giữa các note là quan hệ tri thức.
- **Cách làm thật**: Trong `Indexer.index_file()`, parse `[[wikilink]]` và markdown link nội bộ (`[text](/notes/x.md)`), lưu vào metadata chunk (`links: "a.md,b.md"`). Không cần graph DB.
- **Effort**: ~1–2 ngày.

#### F3. Tương thích Obsidian vault

- **Giá trị**: Cộng đồng Obsidian lớn (claude-obsidian 8.8k⭐, smart-connections 5.2k⭐ đều sống trong hệ sinh thái này).
- **Cách làm thật**: Phần lớn là **cấu hình chứ không phải code**: trỏ `NOTES_DIR` vào vault, bỏ qua thư mục `.obsidian/` khi index, hỗ trợ wikilink (từ F2). Frontmatter OKF tương thích sẵn với Obsidian properties.
- **Effort**: ~1–2 ngày (sau F2).

#### F4. Related notes / Backlinks

- **Giá trị**: Feature được yêu thích nhất ở nhóm PKM-AI (smart-connections 5.2k⭐; reor cũng xây quanh auto-linking).
- **Cách làm thật**: **Tái sử dụng embeddings đã có trong ChromaDB** — query similarity giữa các note, endpoint `GET /related/{filename}` + panel nhỏ trong UI hiện tại. Không cần NER, không cần model mới (báo cáo cũ đề xuất NER tiếng Anh — mâu thuẫn với dự án tiếng Việt).
- **Effort**: ~3–5 ngày.

#### F5. Ingest PDF/DOCX

- **Giá trị**: Mở rộng nguồn tri thức ngoài markdown (khoj, anything-llm đều hỗ trợ).
- **Cách làm thật**: Dùng `markitdown` (163k⭐) convert → ghi file `.md` kèm frontmatter OKF (`type: reference`, `resource:` trỏ file gốc) vào `data/notes/` — pipeline index hiện tại tự xử lý phần còn lại qua watcher.
- **Effort**: ~2–3 ngày.

#### F6. Multi-LLM qua OpenAI-compatible API

- **Giá trị**: Đổi model linh hoạt (LM Studio, vLLM, OpenRouter, cloud khi cần chất lượng).
- **Cách làm thật**: Ollama đã expose endpoint OpenAI-compatible (`/v1/chat/completions`). Refactor `LLMClient` sang chat-completions format (việc này cũng loại bỏ hack strip prefix "Assistant:" hiện tại) + config `LLM_API_KEY` tùy chọn. **Không cần LiteLLM** — thêm dependency nặng không cần thiết cho 1 giao thức chuẩn.
- **Effort**: ~2–3 ngày.

### ⚠️ LÀM SAU, CÓ ĐIỀU KIỆN

#### F7. Knowledge Graph view

- Chỉ có giá trị **sau khi** F2 tạo ra dữ liệu link và người dùng thực sự viết note có link. Với vault hiện tại (6 notes, 0 wikilink) graph sẽ rỗng.
- Cách làm nhẹ: endpoint `GET /graph` trả JSON nodes/links từ metadata F2, render bằng thư viện `force-graph` (1 file JS) **trong UI hiện tại** — không cần migrate React như báo cáo cũ.
- Effort khi đến lúc: ~1 tuần.

#### F8. Retrieval evaluation harness

- Điều kiện tiên quyết cho mọi ý tưởng "advanced chunking": bộ ~20–30 câu hỏi vàng + đo hit-rate/MRR trước-sau. Không có đo lường thì mọi thay đổi chunking chỉ là cảm tính.
- Effort: ~2–3 ngày.

### ❌ KHÔNG LÀM (với lý do)

| Tính năng (báo cáo cũ đề xuất) | Lý do loại |
|---|---|
| React/Electron rewrite | UI hiện tại đáp ứng đủ use-case chat + panel. `reor` — Electron app cùng loại, 8.6k⭐ — đã bị archive. Chi phí 5 tuần cho zero tính năng mới. |
| Multi-user + RBAC, LDAP/AD | Trái định vị single-user local-first; segment team/enterprise đã thuộc anything-llm (62.6k⭐) và Langchain-Chatchat (38.3k⭐). |
| Advanced/semantic chunking | Codebase **đã có** Semantic Sectioning (báo cáo cũ mô tả sai baseline là "split by size"). Chỉ xem xét lại nếu F8 chỉ ra vấn đề đo được. Semantic chunking bằng embedding từng câu cũng quá chậm cho CPU-only. |
| NER concept extraction | Model NER đề xuất là English-only — không dùng được cho note tiếng Việt. F4 đạt cùng mục tiêu bằng embeddings sẵn có. |
| Monetization / "Enterprise-ready" | Ngoài phạm vi một công cụ cá nhân mã nguồn mở ở giai đoạn này. |

---

## 4. Roadmap đề xuất

```
Phase 1 — "MCP + Links" (~2 tuần)
├── F1  MCP server (read-only tools)          2–4 ngày   ★ khác biệt hóa lớn nhất
├── F2  Wikilink/OKF link trong indexer       1–2 ngày
└── F3  Tương thích Obsidian vault            1–2 ngày

Phase 2 — "Knowledge surface" (~2–3 tuần)
├── F4  Related notes + backlinks             3–5 ngày
├── F5  Ingest PDF/DOCX (markitdown)          2–3 ngày
└── F6  Multi-LLM (OpenAI-compatible)         2–3 ngày

Phase 3 — theo nhu cầu thực tế (đo trước, làm sau)
├── F8  Evaluation harness                    2–3 ngày   (làm trước F7 nếu phân vân)
└── F7  Knowledge graph view                  ~1 tuần    (chỉ khi vault đã có link)
```

Tổng Phase 1+2: **~4–5 tuần** effort thực, toàn bộ đều tái sử dụng hạ tầng hiện có (ChromaDB, embeddings, watcher, UI), không thêm framework mới nào ngoài `mcp` SDK và `markitdown`.

## 5. Định vị

> **DevMemory Pro = knowledge base cá nhân cho developer Việt: nhẹ (CPU-only), riêng tư (local), chuẩn mở (OKF markdown), và cắm được vào mọi AI assistant qua MCP.**

Không đua với khoj/anything-llm về bề rộng; thắng ở chiều sâu của niche: tiếng Việt + dev notes + date-aware + MCP.

---

*Nguồn số liệu: GitHub, kiểm chứng trực tiếp ngày 2026-07-05. Đánh giá effort dựa trên codebase tại commit `8f6ad1b`.*
