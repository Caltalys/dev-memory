import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings, logger


class Retriever:
    def __init__(self, indexer):
        self.collection = indexer.collection
        self.embedder = indexer.embedder
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []
        self._bm25 = None
        self._load_corpus()

    def _load_corpus(self):
        """Load toàn bộ documents từ ChromaDB và build BM25 index."""
        try:
            import bm25s
            all_data = self.collection.get(include=["documents"])
            self._corpus = all_data.get("documents") or []
            self._corpus_ids = all_data.get("ids") or []

            if self._corpus:
                corpus_tokens = bm25s.tokenize(self._corpus, stopwords="en")
                self._bm25 = bm25s.BM25()
                self._bm25.index(corpus_tokens)
                logger.info(f"BM25 index built with {len(self._corpus)} documents.")
            else:
                logger.warning("Corpus is empty — BM25 index skipped.")
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}. Falling back to vector-only.")
            self._bm25 = None

    def reload_corpus(self):
        """Reload BM25 corpus (gọi sau khi index lại)."""
        self._load_corpus()

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Tìm kiếm bằng vector embeddings."""
        query_emb = self.embedder.encode([query]).tolist()
        result = self.collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k * 2, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if result["ids"] and result["ids"][0]:
            for i, doc_id in enumerate(result["ids"][0]):
                dist = result["distances"][0][i]
                score = 1.0 - dist  # cosine similarity
                items.append({
                    "id": doc_id,
                    "content": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "score": score,
                    "method": "vector",
                })
        return items

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Tìm kiếm bằng BM25 keyword."""
        if not self._bm25 or not self._corpus:
            return []
        try:
            import bm25s
            query_tokens = bm25s.tokenize([query], stopwords="en")
            results, scores = self._bm25.retrieve(query_tokens, k=min(top_k * 2, len(self._corpus)))

            items = []
            for i in range(results.shape[1]):
                doc_idx = results[0][i]
                score = float(scores[0][i])
                if score <= 0:
                    continue
                items.append({
                    "id": self._corpus_ids[doc_idx] if doc_idx < len(self._corpus_ids) else str(doc_idx),
                    "content": self._corpus[doc_idx],
                    "metadata": {},
                    "score": score,
                    "method": "bm25",
                })
            return items
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        vec_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Fuse kết quả Vector và BM25 bằng Reciprocal Rank Fusion."""
        score_map: dict[str, float] = {}
        meta_map: dict[str, dict] = {}

        for rank, item in enumerate(vec_results):
            doc_id = item["id"]
            score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (k + rank + 1)
            meta_map[doc_id] = item

        # Normalize BM25 scores và add into fusion
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results) or 1.0
            for rank, item in enumerate(bm25_results):
                doc_id = item["id"]
                score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (k + rank + 1)
                if doc_id not in meta_map:
                    meta_map[doc_id] = item

        # Sort by fused score
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
        return [meta_map[doc_id] for doc_id in sorted_ids]

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Hybrid search: Vector + BM25 với Reciprocal Rank Fusion."""
        top_k = top_k or settings.TOP_K

        if self.collection.count() == 0:
            logger.warning("Vector DB is empty. Please run indexer first.")
            return []

        vec_results = self._vector_search(query, top_k)
        bm25_results = self._bm25_search(query, top_k)

        fused = self._reciprocal_rank_fusion(vec_results, bm25_results)
        return fused[:top_k]
