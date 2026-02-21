import os
import hashlib
import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.config import settings, logger


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

    def chunk_text(self, text: str) -> list[str]:
        """Chia text thành chunks, cố gắng giữ nguyên code blocks."""
        lines = text.split('\n')
        chunks = []
        current_chunk: list[str] = []
        current_len = 0
        in_code_block = False

        for line in lines:
            # Track code block boundaries
            if line.strip().startswith('```'):
                in_code_block = not in_code_block

            line_len = len(line)

            # Nếu vượt quá CHUNK_SIZE và không ở trong code block → flush chunk
            if (
                current_len + line_len > settings.CHUNK_SIZE
                and current_chunk
                and not in_code_block
            ):
                chunks.append('\n'.join(current_chunk))
                # Overlap: giữ lại vài dòng cuối
                overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk[:]
                current_chunk = overlap_lines
                current_len = sum(len(l) for l in current_chunk)

            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return [c for c in chunks if c.strip()]

    def index_file(self, filepath: Path):
        """Index một file markdown vào ChromaDB."""
        try:
            post = frontmatter.load(filepath)
            content = post.content
            metadata = dict(post.metadata)
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()

            # Xóa chunks cũ của file này
            existing = self.collection.get(where={"source_file": str(filepath)})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])

            chunks = self.chunk_text(content)
            if not chunks:
                logger.warning(f"No content in {filepath.name}, skipping.")
                return

            ids = []
            docs = []
            metas = []

            for i, chunk in enumerate(chunks):
                ids.append(f"{file_id}_{i}")
                docs.append(chunk)
                tags = metadata.get("tags", [])
                metas.append({
                    "source_file": str(filepath),
                    "filename": filepath.name,
                    "tags": ", ".join(tags) if isinstance(tags, list) else str(tags),
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
            logger.info(f"✓ Indexed: {filepath.name} ({len(docs)} chunks)")

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
