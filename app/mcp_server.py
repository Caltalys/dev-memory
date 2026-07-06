"""DevMemory MCP Server — expose knowledge base cho AI assistant qua MCP.

Chạy standalone qua stdio transport:

    python -m app.mcp_server

Cấu hình cho Claude Desktop / Claude Code / Cursor (ví dụ):

    {
      "mcpServers": {
        "devmemory": {
          "command": "/path/to/venv/bin/python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "/path/to/dev-memory"
        }
      }
    }

Chỉ cung cấp tool read-only (search/ask/list). Tool ghi note sẽ được
bổ sung sau kèm validation frontmatter OKF — xem docs/ROADMAP_ANALYSIS_2026-07.md.
"""

import frontmatter
from mcp.server.fastmcp import FastMCP

from app.config import settings, logger, RESERVED_FILENAMES

mcp = FastMCP("devmemory")

# Lazy singletons — load model embedding (~2GB) chỉ khi tool đầu tiên được gọi,
# để handshake tools/list phản hồi ngay lập tức.
_rag = {}


def _get_rag():
    if "retriever" not in _rag:
        logger.info("MCP: initializing indexer/retriever (first tool call)...")
        from app.indexer import Indexer
        from app.retriever import Retriever

        indexer = Indexer()
        _rag["indexer"] = indexer
        _rag["retriever"] = Retriever(indexer)
    return _rag["retriever"]


def _format_chunk(i: int, chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    header = (
        f"[{i}] {meta.get('filename', 'unknown')} "
        f"({meta.get('type', 'unknown')}, {meta.get('timestamp', '')}) "
        f"— section: {meta.get('section_title', '')}"
    )
    content = chunk.get("content", "").strip()
    if len(content) > 800:
        content = content[:800].rsplit("\n", 1)[0] + "\n[...truncated]"
    return f"{header}\n{content}"


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> str:
    """Tìm kiếm trong DevMemory knowledge base bằng hybrid search (vector + BM25).

    Hỗ trợ tiếng Việt và lọc thời gian tự nhiên trong câu truy vấn
    (ví dụ: "lỗi pip tuần trước"). Trả về các đoạn note liên quan nhất
    kèm tên file nguồn, loại note và ngày.

    Args:
        query: Câu truy vấn (ngôn ngữ tự nhiên, tiếng Việt hoặc tiếng Anh).
        top_k: Số kết quả tối đa (mặc định 5).
    """
    retriever = _get_rag()
    chunks = retriever.retrieve(query, top_k=top_k)
    if not chunks:
        return "Không tìm thấy kết quả nào trong knowledge base."
    return "\n\n---\n\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool()
def ask(question: str, top_k: int = 3) -> str:
    """Hỏi DevMemory và nhận câu trả lời đã tổng hợp qua RAG pipeline.

    Retrieve các đoạn note liên quan rồi để LLM local (Ollama) tổng hợp
    câu trả lời kèm trích dẫn nguồn. Chậm hơn search_notes vì phải chạy
    LLM; dùng search_notes nếu chỉ cần đoạn trích thô.

    Args:
        question: Câu hỏi (ngôn ngữ tự nhiên).
        top_k: Số chunk context đưa vào LLM (mặc định 3).
    """
    from app.llm import llm_client

    retriever = _get_rag()
    chunks = retriever.retrieve(question, top_k=top_k)
    if not chunks:
        return "Không tìm thấy thông tin liên quan trong knowledge base."
    return llm_client.ask(question, chunks)


@mcp.tool()
def related_notes(filename: str, top_k: int = 5) -> str:
    """Tìm các note liên quan đến một note: outgoing links, backlinks, và
    note tương đồng ngữ nghĩa (dựa trên embedding có sẵn).

    Args:
        filename: Tên file note, ví dụ "sqlalchemy-n-plus-1.md".
        top_k: Số related note tối đa (mặc định 5).
    """
    retriever = _get_rag()
    conn = retriever.get_note_connections(filename, top_k=top_k)
    lines = [f"Kết nối của {filename}:"]
    lines.append(
        "- Outgoing links: " + (", ".join(conn["outgoing"]) if conn["outgoing"] else "(không có)")
    )
    lines.append(
        "- Backlinks: " + (", ".join(conn["backlinks"]) if conn["backlinks"] else "(không có)")
    )
    if conn["related"]:
        lines.append("- Related (semantic):")
        lines.extend(f"    {r['filename']} (score: {r['score']})" for r in conn["related"])
    else:
        lines.append("- Related (semantic): (không có)")
    return "\n".join(lines)


@mcp.tool()
def list_notes(tag: str = "", limit: int = 50) -> str:
    """Liệt kê các note trong knowledge base kèm metadata OKF.

    Đọc trực tiếp frontmatter từ thư mục notes (không cần vector DB),
    nên phản hồi nhanh và luôn phản ánh trạng thái file mới nhất.

    Args:
        tag: Lọc theo tag (để trống = tất cả).
        limit: Số note tối đa (mặc định 50).
    """
    rows = []
    md_files = sorted(
        f for f in settings.NOTES_DIR.rglob("*.md")
        if f.name.lower() not in RESERVED_FILENAMES
        and not any(part.startswith(".") for part in f.relative_to(settings.NOTES_DIR).parts)
    )
    for f in md_files:
        try:
            post = frontmatter.load(f)
        except Exception:
            continue
        tags = post.metadata.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        if tag and tag not in tags:
            continue
        rows.append(
            f"- {f.relative_to(settings.NOTES_DIR)} "
            f"(type: {post.metadata.get('type', '?')}, "
            f"tags: {', '.join(str(t) for t in tags) or '—'}, "
            f"timestamp: {post.metadata.get('timestamp', '?')})"
        )
        if len(rows) >= limit:
            break

    if not rows:
        return f"Không có note nào{f' với tag [{tag}]' if tag else ''}."
    return f"Tổng: {len(rows)} note\n" + "\n".join(rows)


if __name__ == "__main__":
    mcp.run()
