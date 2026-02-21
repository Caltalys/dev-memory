---
tags: [llm, rag, prompt-engineering, ollama, python]
project: "DevMemory Pro"
date: 2026-02-21
---

# Tối ưu System Prompt để giảm Hallucination trong RAG Pipeline

## Bối cảnh (Context)
Dự án DevMemory Pro sử dụng mô hình `qwen2.5:1.5b` qua Ollama kết hợp với Hybrid Search (Vector + BM25) để truy vấn knowledge base. Stack: FastAPI + ChromaDB + sentence-transformers.

## Vấn đề (Problem)
Model trả về câu trả lời "thừa thãi" và lẫn lộn thông tin không liên quan khi truy vấn.

Ví dụ: Hỏi "tôi đã xử lý lỗi pip nào" → Model trả về cả code SQLAlchemy (từ note khác) và tự bịa ra các tùy chọn cài đặt `pip` không tồn tại (hallucination).

**Nguyên nhân gốc rễ:**
1. Model nhỏ (1.5B) không tự biết lọc thông tin không liên quan trong context.
2. System prompt cũ có lệnh "Ưu tiên hiển thị code snippet nếu có" → khiến model lôi **mọi** đoạn code trong context ra hiển thị bất kể liên quan hay không.
3. `_build_prompt` dùng nhãn `CONTEXT TỪ KNOWLEDGE BASE:` không khớp với system prompt → model không nhận dạng cấu trúc tốt.

## Giải pháp (Solution)

**1. Đồng bộ nhãn giữa `system_prompt` và `_build_prompt`:**

```python
# _build_prompt: dùng nhãn bracket rõ ràng
return (
    f"[CONTEXT]\n{context_text}"
    f"{history_section}\n\n"
    f"[CÂU HỎI]: {query}\n\n"
    f"[TRẢ LỜI] (chỉ dựa vào context trên, bỏ qua nếu không liên quan):"
)
```

**2. Viết lại system_prompt với nguyên tắc lọc ngữ cảnh rõ ràng:**

```python
self.system_prompt = (
    "Bạn là trợ lý tra cứu knowledge base cá nhân của một developer."
    "NGUYÊN TẮC:"
    "- Chỉ sử dụng nội dung trong [CONTEXT] để trả lời [CÂU HỎI]."
    "- Mỗi đoạn [CONTEXT] có tên file nguồn ở đầu. "
    "Nếu nội dung đoạn đó KHÔNG liên quan đến [CÂU HỎI], "
    "hãy bỏ qua hoàn toàn, kể cả code trong đó."
    "- Nếu không có đoạn [CONTEXT] nào liên quan: trả lời chính xác "
    "'Không tìm thấy thông tin liên quan trong knowledge base.'"
    "- Không thêm thông tin hoặc lệnh ngoài phạm vi [CONTEXT]."
    "- Ngắn gọn, chính xác, trích dẫn nguồn ở cuối nếu có."
    "FORMAT: Markdown."
)
```

## Bài học (Lesson Learned)

- **Nhãn (labels) trong prompt phải nhất quán** giữa system prompt và user prompt. Model sẽ "bám" vào các nhãn cố định như `[CONTEXT]`, `[CÂU HỎI]` để phân biệt cấu trúc.
- **Không nên có lệnh chung chung như "ưu tiên code"** trong RAG context vì model nhỏ sẽ hiểu là lấy code từ *bất kỳ* đoạn context nào.
- **Lệnh fallback phải được viết cứng** (ví dụ: "trả lời chính xác: 'Không tìm thấy...'") để tránh model paraphrase và thêm thông tin không cần thiết.
- Model `1.5B` có giới hạn về reasoning. Nếu cần kết quả tốt hơn, nâng cấp lên `qwen2.5:7b` hoặc `llama3.1:8b`.

## Tham khảo (References)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Prompt Engineering Guide - RAG](https://www.promptingguide.ai/techniques/rag)
