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

        self.system_prompt = (
            "Bạn là trợ lý tra cứu knowledge base cá nhân của một developer.\n"
            "NGUYÊN TẮC:\n"
            "- Chỉ sử dụng nội dung trong [CONTEXT] để trả lời [CÂU HỎI].\n"
            "- Mỗi đoạn [CONTEXT] có dạng '[n] tên-file (loại, ngày)'. "
            "Nếu đoạn nào KHÔNG liên quan đến [CÂU HỎI], "
            "hãy bỏ qua hoàn toàn, kể cả code trong đó.\n"
            "- [HOẠT ĐỘNG TRƯỚC] chỉ để hiểu ngữ cảnh hội thoại, "
            "KHÔNG dùng làm nguồn trả lời.\n"
            "- Nếu không có đoạn [CONTEXT] nào liên quan: trả lời chính xác "
            "'Không tìm thấy thông tin liên quan trong knowledge base.'\n"
            "- Không thêm thông tin hoặc lệnh ngoài phạm vi [CONTEXT].\n"
            "- Ngắn gọn, chính xác. Cuối câu trả lời, liệt kê nguồn theo dạng "
            "'Nguồn: [n] tên-file.md'.\n"
            "FORMAT: Markdown."
        )

        # Options tối ưu cho CPU, ngăn hallucination
        self._options = {
            "temperature": 0.2,
            "top_p": 0.85,
            "num_ctx": settings.NUM_CTX,
            "num_predict": settings.MAX_TOKENS,
            "repeat_penalty": 1.1,
            "stop": ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"],
        }

    @staticmethod
    def _truncate_content(content: str, limit: int = 600) -> str:
        """Cắt content tại ranh giới dòng và tự đóng code fence nếu bị cắt dở.

        <p>Tránh cắt ngang ``` khiến phần còn lại của response bị coi là code
        block, làm model "bịa" tiếp phần code đã bị cắt.
        """
        content = content.strip()
        if len(content) <= limit:
            return content

        truncated = content[:limit].rsplit("\n", 1)[0]
        if truncated.count("```") % 2 == 1:
            truncated += "\n```"
        return truncated

    @staticmethod
    def _truncate_history(history: str, limit: int = 200) -> str:
        """Cắt history tại ranh giới lượt hội thoại (User:/Assistant:)."""
        history = history.strip()
        if len(history) <= limit:
            return history

        truncated = history[:limit]
        for marker in ("\nUser:", "\nAssistant:"):
            idx = truncated.rfind(marker)
            if idx > 0:
                return truncated[:idx]
        return truncated

    def _build_prompt(self, query: str, chunks: list, history: str = "") -> str:
        """Xây dựng prompt ngắn gọn — truncate chunk để tiết kiệm context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("filename", "unknown")
            note_type = meta.get("type", "unknown")
            ts = meta.get("timestamp", "")
            content = self._truncate_content(chunk.get("content", ""))
            context_parts.append(f"[{i}] {source} ({note_type}, {ts}):\n{content}")
        context_text = "\n\n".join(context_parts)

        # Chỉ dùng history nếu thực sự có, giới hạn rất ngắn
        history_section = ""
        if history:
            history_short = self._truncate_history(history)
            if history_short:
                history_section = f"\n\n[HOẠT ĐỘNG TRƯỚC] (tóm tắt):\n{history_short}"

        return (
            f"[CONTEXT]\n{context_text}"
            f"{history_section}\n\n"
            f"[CÂU HỎI]: {query}\n\n"
            f"[TRẢ LỜI] (chỉ dựa vào context trên, bỏ qua nếu không liên quan):"
        )

    def ask_stream(self, query: str, chunks: List[Dict], history: str = "") -> Generator[str, None, None]:
        """Streaming response — yield từng token ngay khi nhận được từ LLM.

        <p>Provider chọn qua {@code LLM_PROVIDER}: {@code ollama} (native API,
        mặc định — giữ được num_ctx/repeat_penalty) hoặc {@code openai}
        (OpenAI-compatible chat completions: LM Studio, vLLM, OpenRouter,
        Ollama {@code /v1}, ...).
        """
        prompt = self._build_prompt(query, chunks, history)
        try:
            if settings.LLM_PROVIDER == "openai":
                yield from self._stream_openai(prompt)
            else:
                yield from self._stream_ollama(prompt)
        except httpx.ConnectError:
            logger.error(f"Cannot connect to LLM at {self.base_url}.")
            yield f"⚠️ Lỗi kết nối: Không thể kết nối LLM ({self.base_url})."
        except httpx.TimeoutException:
            logger.warning("LLM streaming timeout.")
            yield "\n⚠️ Timeout: Model phản hồi quá chậm."
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"⚠️ Lỗi: {e}"

    def _stream_ollama(self, prompt: str) -> Generator[str, None, None]:
        """Stream qua Ollama native API ({@code /api/generate})."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": True,
            "options": self._options,
        }

        first_token = True
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
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                if token:
                    if first_token:
                        # Strip prefix "Assistant:" hoặc ký tự trắng đầu
                        token = token.lstrip()
                        for prefix in ("Assistant:", "A:"):
                            if token.startswith(prefix):
                                token = token[len(prefix):].lstrip()
                        first_token = False
                    yield token
                if data.get("done"):
                    break

    def _openai_base(self) -> str:
        """Chuẩn hoá base URL cho OpenAI-compatible API (đảm bảo có /v1)."""
        return self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"

    def _stream_openai(self, prompt: str) -> Generator[str, None, None]:
        """Stream qua OpenAI-compatible chat completions API.

        <p>Lưu ý: các option Ollama-specific ({@code num_ctx},
        {@code repeat_penalty}) không tồn tại trong giao thức này.
        """
        headers = {}
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": self._options["temperature"],
            "top_p": self._options["top_p"],
            "max_tokens": self._options["num_predict"],
        }

        with httpx.stream(
            "POST",
            f"{self._openai_base()}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                token = (choices[0].get("delta") or {}).get("content")
                if token:
                    yield token

    def ask(self, query: str, chunks: List[Dict], history: str = "") -> str:
        """Non-streaming fallback — dùng cho internal calls (save to memory, etc.)."""
        return "".join(self.ask_stream(query, chunks, history))

    def check_health(self) -> bool:
        try:
            if settings.LLM_PROVIDER == "openai":
                headers = {}
                if settings.LLM_API_KEY:
                    headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
                response = httpx.get(f"{self._openai_base()}/models", headers=headers, timeout=5.0)
            else:
                response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
llm_client = LLMClient()
