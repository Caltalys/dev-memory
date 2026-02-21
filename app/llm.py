import httpx
import json
from typing import List, Dict, Generator
from app.config import settings, logger


class LLMClient:
    def __init__(self):
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL
        self.timeout = 120.0
        self.max_retries = 2

        self.system_prompt = """Bạn là trợ lý tra cứu knowledge base của một developer.

QUY TẮc BẮt BUỘC:
1. Chỉ trả lời MỘT LẦN. Đóng gói toàn bộ thông tin vào một câu trả lời duy nhất.
2. TUYỆT ĐỐI KHÔNG tự viết tiếp User:, Human:, Q:, A: sau câu trả lời.
3. Chỉ dùng context được cung cấp. Nếu không có thông tin, nói “không tìm thấy trong knowledge base”.
4. Ưu tiên hiển thị code snippet nếu có.
5. Ngắn gọn, chính xác, không lặp lại.

FORMAT: Markdown. Nguồn ở cuối nếu có."""

        # Options tối ưu cho CPU, ngăn hallucination
        self._options = {
            "temperature": 0.2,      # Thấp hơn → ít suy diễn hơn
            "top_p": 0.85,
            "num_ctx": int(settings.__dict__.get("NUM_CTX", 2048)),
            "num_predict": int(settings.__dict__.get("MAX_TOKENS", 512)),
            "repeat_penalty": 1.3,   # Tăng lên để chống lặp
            # QUAN TRọNG: Dừng lại khi gặp các tín hiệu bắt đầu fake conversation
            "stop": ["User:", "Human:", "\nUser", "\nHuman", "Q:", "\nQ:", "Assistant:"],
        }

    def _build_prompt(self, query: str, chunks: list, history: str = "") -> str:
        """Xây dựng prompt ngắn gọn — truncate chunk để tiết kiệm context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("filename", "unknown")
            content = chunk.get("content", "")[:600].strip()
            context_parts.append(f"[{i}] {source}:\n{content}")
        context_text = "\n\n".join(context_parts)

        # Chỉ dùng history nếu thực sự có, giới hạn rất ngắn
        history_section = ""
        if history:
            # Lấy tối đa 200 ky tự của history để không expand context
            history_short = history.strip()[:200]
            history_section = f"\n\n[HOẠT ĐỘNG TRƯỚC] (tóm tắt):\n{history_short}"

        return (
            f"CONTEXT TỪ KNOWLEDGE BASE:\n{context_text}"
            f"{history_section}\n\n"
            f"CÂU HỊI: {query}\n\n"
            f"TRẢ LỜI (chỉ dựa vào context trên, không thêm giả định):"
        )

    def ask_stream(self, query: str, chunks: List[Dict], history: str = "") -> Generator[str, None, None]:
        """Streaming response — yield từng token ngay khi nhận được từ Ollama."""
        prompt = self._build_prompt(query, chunks, history)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": True,   # ← KEY: streaming mode
            "options": self._options,
        }

        first_token = True
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            if first_token:
                                # Strip prefix "Assistant:" hoặc ký tự trắng đầu
                                token = token.lstrip()
                                # Nếu token bắt đầu bằng "Assistant:", bỏ qua
                                for prefix in ("Assistant:", "A:"):
                                    if token.startswith(prefix):
                                        token = token[len(prefix):].lstrip()
                                first_token = False
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama.")
            yield "⚠️ Lỗi kết nối: Không thể kết nối Ollama."
        except httpx.TimeoutException:
            logger.warning("LLM streaming timeout.")
            yield "\n⚠️ Timeout: Model phản hồi quá chậm."
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"⚠️ Lỗi: {e}"

    def ask(self, query: str, chunks: List[Dict], history: str = "") -> str:
        """Non-streaming fallback — dùng cho internal calls (save to memory, etc.)."""
        return "".join(self.ask_stream(query, chunks, history))

    def check_health(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
llm_client = LLMClient()
