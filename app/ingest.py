"""Ingest tài liệu ngoài markdown (PDF/DOCX/...) vào knowledge base.

Convert file nguồn sang Markdown bằng `markitdown`, gắn frontmatter OKF
(`type: reference`, `resource:` trỏ về file gốc) rồi ghi vào NOTES_DIR —
pipeline index hiện có (watcher/indexer) tự xử lý phần còn lại.
"""

import datetime
import re
import unicodedata
from pathlib import Path

from app.config import settings, logger

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}


def _slugify(name: str) -> str:
    """Chuyển tên file gốc thành slug an toàn cho tên file markdown.

    <p>Khử dấu tiếng Việt (NFKD), thay ký tự không phải chữ/số bằng gạch
    ngang, gộp gạch liên tiếp.
    """
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return name or "document"


def convert_to_markdown(src_path: Path) -> str:
    """Convert file nguồn sang markdown text bằng markitdown.

    @param src_path Đường dẫn file nguồn (pdf/docx/...).
    @return Nội dung markdown.
    @raises ValueError nếu định dạng không được hỗ trợ hoặc convert thất bại.
    """
    if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Định dạng không được hỗ trợ: {src_path.suffix}. "
            f"Hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    from markitdown import MarkItDown

    result = MarkItDown().convert(str(src_path))
    text = (result.text_content or "").strip()
    if not text:
        raise ValueError(f"Không trích xuất được nội dung từ {src_path.name}")
    return text


def ingest_file(src_path: Path, original_name: str = None) -> Path:
    """Ingest một file tài liệu: convert → frontmatter OKF → ghi vào NOTES_DIR.

    @param src_path      Đường dẫn file nguồn trên đĩa (có thể là temp file).
    @param original_name Tên file gốc do người dùng upload (dùng cho title/slug);
                         mặc định lấy từ src_path.
    @return Đường dẫn file markdown đã tạo trong NOTES_DIR.
    """
    original_name = original_name or src_path.name
    title = Path(original_name).stem
    text = convert_to_markdown(src_path)

    today = datetime.date.today().isoformat()
    slug = _slugify(title)
    dest = settings.NOTES_DIR / f"{today}-{slug}.md"
    # Tránh ghi đè nếu trùng tên
    counter = 1
    while dest.exists():
        dest = settings.NOTES_DIR / f"{today}-{slug}-{counter}.md"
        counter += 1

    heading = "" if text.lstrip().startswith("#") else f"# {title}\n\n"
    content = (
        "---\n"
        f"type: reference\n"
        f"tags: [imported]\n"
        f'project: ""\n'
        f"resource: {original_name}\n"
        f"timestamp: {today}T00:00:00Z\n"
        "---\n\n"
        f"{heading}{text}\n"
    )
    dest.write_text(content, encoding="utf-8")
    logger.info(f"📥 Ingested {original_name} → {dest.name} ({len(text)} chars)")
    return dest
