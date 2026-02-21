import chromadb
import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from sentence_transformers import SentenceTransformer
from app.config import settings, logger

_VI_STOPWORDS = [
    "là", "và", "hoặc", "nhưng", "mà", "thì", "của", "trong", "với", "cho",
    "để", "nên", "không", "có", "một", "được", "này", "nào", "đó", "các",
    "những", "khi", "nếu", "vì", "từ", "ra", "vẫn", "lên", "về", "bởi",
    "nên", "hơn", "nhất", "hãy", "cũng", "đã", "sẽ", "đang", "rất", "như",
]


def _extract_date_filter(query: str) -> dict | None:
    """Trích xuất ChromaDB where-clause từ câu hỏi mang tính thời gian.

    <p>Hỗ trợ các dạng:
    <ul>
        <li>Tương đối hiện tại: "hôm nay", "today"</li>
        <li>Tương đối ngày qua: "hôm qua", "yesterday"</li>
        <li>Khoảng thời gian: "2 tháng trước", "3 tuần trước", "10 ngày trước", "1 năm trước"</li>
        <li>Cụ thể dd/mm/yyyy: "20/10/2025" hoặc "20-10-2025"</li>
        <li>Cụ thể ISO yyyy-mm-dd: "2024-11-23"</li>
        <li>Có tiền tố: "ngày 20/10/2025"</li>
    </ul>

    @param query Câu hỏi từ người dùng.
    @return ChromaDB {@code where} dict nếu phát hiện từ khóa thời gian, {@code None} nếu không có.
    """
    q = query.lower()
    today = date.today()

    # ── Tương đối: hôm nay / hôm qua ──────────────────────────────────────────
    if any(kw in q for kw in ["hôm nay", "today", "hom nay"]):
        return {"date": {"$eq": str(today)}}
    if any(kw in q for kw in ["hôm qua", "yesterday", "hom qua"]):
        return {"date": {"$eq": str(today - timedelta(days=1))}}

    # ── Khoảng thời gian: "X (ngày|tuần|tháng|năm) trước" ────────────────────
    m = re.search(
        r"(\d+)\s*(ngày|tháng|tuần|năm|ngay|thang|tuan|nam|day|days|week|weeks|month|months|year|years)\s*(trước|ago)",
        q,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in ("ngày", "ngay", "day", "days"):
            since = today - timedelta(days=n)
        elif unit in ("tuần", "tuan", "week", "weeks"):
            since = today - timedelta(weeks=n)
        elif unit in ("tháng", "thang", "month", "months"):
            since = today - relativedelta(months=n)
        else:  # năm / nam / year / years
            since = today - relativedelta(years=n)
        return {"date": {"$gte": str(since), "$lte": str(today)}}

    # ── Ngày cụ thể: dd/mm/yyyy hoặc dd-mm-yyyy (có hoặc không có "ngày") ────
    m = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", q)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return {"date": {"$eq": f"{y}-{mo:02d}-{d:02d}"}}

    # ── Ngày cụ thể: yyyy-mm-dd ────────────────────────────────────────────────
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", q)
    if m:
        return {"date": {"$eq": f"{m.group(1)}-{m.group(2)}-{m.group(3)}"}}

    return None


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
            all_data = self.collection.get(include=["documents", "metadatas"])
            self._corpus = all_data.get("documents") or []
            self._corpus_ids = all_data.get("ids") or []
            self._corpus_metadata = all_data.get("metadatas") or []

            if self._corpus:
                corpus_tokens = bm25s.tokenize(self._corpus, stopwords=_VI_STOPWORDS)
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

    def _vector_search(self, query: str, top_k: int, where: dict | None = None) -> list[dict]:
        """Tìm kiếm bằng vector embeddings.
        
        @param query Câu hỏi cần tìm kiếm.
        @param top_k Số lượng kết quả tối đa.
        @param where Bộ lọc metadata ChromaDB, {@code None} nếu không lọc.
        @return Danh sách các chunk phù hợp.
        """
        query_emb = self.embedder.encode([query]).tolist()

        query_kwargs = dict(
            query_embeddings=query_emb,
            n_results=min(top_k * 2, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        result = self.collection.query(**query_kwargs)

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
            query_tokens = bm25s.tokenize([query], stopwords=_VI_STOPWORDS)
            results, scores = self._bm25.retrieve(query_tokens, k=min(top_k * 2, len(self._corpus)))

            items = []
            for i in range(results.shape[1]):
                doc_idx = results[0][i]
                score = float(scores[0][i])
                if score <= 0:
                    continue
                meta = (
                    self._corpus_metadata[doc_idx]
                    if doc_idx < len(self._corpus_metadata)
                    else {}
                )
                items.append({
                    "id": self._corpus_ids[doc_idx] if doc_idx < len(self._corpus_ids) else str(doc_idx),
                    "content": self._corpus[doc_idx],
                    "metadata": meta,
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
        """Hybrid search: Vector + BM25 với Reciprocal Rank Fusion.
        
        <p>Tự động áp dụng bộ lọc ngày nếu câu hỏi chứa từ khóa thời gian
        ("hôm nay", "hôm qua", v.v.). Khi date filter được áp dụng, chỉ
        tìm kiếm vector (BM25 không hỗ trợ metadata filter).
        
        @param query Câu hỏi cần tìm kiếm.
        @param top_k Số lượng kết quả tối đa, mặc định theo cấu hình {@code settings.TOP_K}.
        @return Danh sách các chunk phù hợp, sắp xếp theo độ liên quan.
        """
        top_k = top_k or settings.TOP_K

        if self.collection.count() == 0:
            logger.warning("Vector DB is empty. Please run indexer first.")
            return []

        # Phát hiện truy vấn thời gian và áp dụng date filter
        where_filter = _extract_date_filter(query)

        if where_filter:
            logger.info(f"Date filter detected: {where_filter}")

        vec_results = self._vector_search(query, top_k, where=where_filter)

        # BM25 không hỗ trợ metadata filter → bỏ qua khi có date filter
        bm25_results = self._bm25_search(query, top_k) if not where_filter else []

        fused = self._reciprocal_rank_fusion(vec_results, bm25_results)
        return fused[:top_k]
