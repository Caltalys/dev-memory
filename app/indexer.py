import os
import re
import hashlib
import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.config import settings, logger


# ─── Section type classifier ──────────────────────────────────────────────────

_SECTION_TYPES = {
    "symptom":   ["lỗi", "triệu chứng", "bug", "error", "vấn đề", "problem", "issue", "exception"],
    "root_cause": ["nguyên nhân", "tại sao", "cause", "reason", "why"],
    "solution":  ["giải pháp", "cách fix", "solution", "xử lý", "khắc phục", "resolve", "fix"],
    "code":      ["code", "snippet", "ví dụ", "example", "demo", "implementation"],
    "lesson":    ["bài học", "lưu ý", "lesson", "note", "tip", "takeaway", "kết luận"],
}


def _classify_section(title: str) -> str:
    """Phân loại section dựa trên tiêu đề heading.

    @param title Tiêu đề section (không bao gồm ký tự #).
    @return Chuỗi loại section, một trong: {@code symptom}, {@code root_cause},
            {@code solution}, {@code code}, {@code lesson}, {@code general}.
    """
    t = title.lower()
    for section_type, keywords in _SECTION_TYPES.items():
        if any(k in t for k in keywords):
            return section_type
    return "general"


# ─── Chunking strategies ──────────────────────────────────────────────────────

def _split_by_headers(content: str) -> list[dict]:
    """Tách nội dung markdown thành các section theo heading cấp 2 (##).

    <p>Mỗi section được coi là một đơn vị tri thức độc lập. Section đầu tiên
    (nội dung trước heading đầu tiên) được coi là phần giới thiệu với loại
    {@code general}.

    @param content Nội dung markdown đã loại bỏ frontmatter.
    @return Danh sách dict chứa {@code title}, {@code content}, {@code type}.
    """
    # Tách theo dòng bắt đầu bằng ## (giữ nguyên phần trước heading đầu)
    parts = re.split(r'\n(?=##\s)', content)
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split('\n')
        first_line = lines[0]

        if first_line.startswith('##'):
            title = re.sub(r'^#+\s*', '', first_line).strip()
            body = '\n'.join(lines[1:]).strip()
        else:
            # Phần giới thiệu trước heading đầu tiên
            title = "Giới thiệu"
            body = part

        if not body:
            continue

        # Nếu section quá dài, fallback sang fixed-size chunking cho section đó
        sub_chunks = _split_large_section(title, body)
        sections.extend(sub_chunks)

    return sections


def _split_large_section(title: str, body: str) -> list[dict]:
    """Chia một section lớn thành các sub-chunk nếu vượt quá ngưỡng.

    <p>Tôn trọng ranh giới code block khi cắt. Sử dụng {@code settings.CHUNK_SIZE}
    làm ngưỡng tối đa cho mỗi sub-chunk.

    @param title  Tiêu đề của section cha.
    @param body   Nội dung thuần của section (không có dòng heading).
    @return Danh sách dict, mỗi phần tử đại diện cho một sub-chunk.
    """
    section_type = _classify_section(title)
    max_size = settings.CHUNK_SIZE * 2  # Generous limit: 2× chunk_size

    if len(body) <= max_size:
        return [{"title": title, "content": f"## {title}\n{body}", "type": section_type}]

    # Cắt tại ranh giới dòng, tôn trọng code block
    lines = body.split('\n')
    chunks = []
    current: list[str] = []
    current_len = 0
    in_code_block = False
    part_idx = 0

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block

        if current_len + len(line) > max_size and current and not in_code_block:
            sub_title = f"{title} (phần {part_idx + 1})"
            chunks.append({
                "title": sub_title,
                "content": f"## {sub_title}\n" + '\n'.join(current),
                "type": section_type,
            })
            current = []
            current_len = 0
            part_idx += 1

        current.append(line)
        current_len += len(line)

    if current:
        sub_title = f"{title} (phần {part_idx + 1})" if part_idx > 0 else title
        chunks.append({
            "title": sub_title,
            "content": f"## {sub_title}\n" + '\n'.join(current),
            "type": section_type,
        })

    return chunks


# ─── Indexer ──────────────────────────────────────────────────────────────────

class Indexer:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name="dev_notes_pro",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Indexer initialized. Current chunks: {self.collection.count()}"
        )

    def index_file(self, filepath: Path):
        """Index một file markdown vào ChromaDB theo chiến lược Semantic Sectioning.

        <p>Mỗi heading cấp 2 ({@code ##}) được index thành một chunk riêng biệt
        kèm metadata {@code section_type} để hỗ trợ filter theo loại nội dung.

        @param filepath Đường dẫn tuyệt đối đến file markdown cần index.
        """
        try:
            post = frontmatter.load(filepath)
            content = post.content
            metadata = dict(post.metadata)
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()

            # Xóa chunks cũ của file này
            existing = self.collection.get(where={"source_file": str(filepath)})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])

            sections = _split_by_headers(content)
            if not sections:
                logger.warning(f"No content in {filepath.name}, skipping.")
                return

            ids = []
            docs = []
            metas = []
            tags = metadata.get("tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            for i, sec in enumerate(sections):
                ids.append(f"{file_id}_{i}")
                docs.append(sec["content"])
                metas.append({
                    "source_file": str(filepath),
                    "filename": filepath.name,
                    "section_title": sec["title"],
                    "section_type": sec["type"],
                    "tags": tags_str,
                    "project": str(metadata.get("project", "unknown")),
                    "date": str(metadata.get("date", "")),
                })

            embeddings = self.embedder.encode(docs, show_progress_bar=False).tolist()
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metas,
            )
            logger.info(
                f"✓ Indexed: {filepath.name} "
                f"({len(docs)} sections: {[s['type'] for s in sections]})"
            )

        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")

    def index_all(self):
        """Index tất cả file markdown trong NOTES_DIR."""
        md_files = list(settings.NOTES_DIR.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files.")
        for f in md_files:
            if f.name.lower() != "template.md":
                self.index_file(f)
        logger.info(f"✅ Indexing complete. Total chunks: {self.collection.count()}")


if __name__ == "__main__":
    idx = Indexer()
    idx.index_all()
