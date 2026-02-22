import chromadb
from datetime import date
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

    <p>Sử dụng thư viện {@code dateparser} để hỗ trợ 200+ ngôn ngữ và các dạng
    biểu đạt tự nhiên, bao gồm:
    <ul>
        <li>Tương đối: "hôm nay", "hôm qua", "today", "yesterday"</li>
        <li>Khoảng thời gian: "2 tháng trước", "tuần trước", "3 weeks ago",
            "hai tháng trước" (hỗ trợ số chữ)</li>
        <li>Ngày cụ thể: "20/10/2025", "2024-11-23", "ngày 20 tháng 10 năm 2025"</li>
    </ul>

    <p>Áp dụng bộ lọc từ khóa để tránh false positive (ví dụ: "timeout 30 giây").

    @param query Câu hỏi từ người dùng, có thể bằng bất kỳ ngôn ngữ nào.
    @return ChromaDB {@code where} dict nếu phát hiện biểu đạt thời gian,
            {@code None} nếu không có hoặc parse thất bại.
    """
    import datetime
    import dateparser

    q = query.lower()
    today = date.today()

    # ── Keyword gate: tránh false positive ────────────────────────────────────
    # Chỉ thử parse khi có từ khóa thời gian rõ ràng
    TIME_KEYWORDS = [
        "hôm nay", "hôm qua", "hom nay", "hom qua",
        "today", "yesterday",
        "trước", "ago", "last ",
        "tuần", "tháng", "năm", "ngày",
        "week", "month", "year", "day",
    ]
    if not any(kw in q for kw in TIME_KEYWORDS):
        return None

    # ── Xác định loại biểu đạt: range hay exact ───────────────────────────────
    RANGE_KEYWORDS = [
        "trước", "ago", "last ", "qua",
        "tuần trước", "tháng trước", "năm trước",
        "last week", "last month", "last year",
    ]
    is_range = any(kw in q for kw in RANGE_KEYWORDS)

    # ── Parse bằng dateparser ─────────────────────────────────────────────────
    try:
        parsed = dateparser.parse(
            query,
            settings={
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": False,
                "RELATIVE_BASE": datetime.datetime.now(),
            },
        )
    except Exception:
        parsed = None

    if not parsed:
        return None

    parsed_date_str = str(parsed.date())

    if is_range:
        # Range query: từ ngày đã parse đến hôm nay
        return {"date": {"$gte": parsed_date_str, "$lte": str(today)}}
    else:
        # Exact query
        return {"date": {"$eq": parsed_date_str}}


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
