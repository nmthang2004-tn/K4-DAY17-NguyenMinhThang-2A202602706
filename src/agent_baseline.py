from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A - within-session memory only.

    Requirements:
    - Within-session memory only
    - No persistent User.md
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # Optionally initialize a real LangChain/LangGraph agent
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        # Use offline path
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Simple offline behavior.

        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """
        # Create session if missing
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Add user message
        session.messages.append({"role": "user", "content": message})

        # Estimate prompt tokens
        prompt_text = "\n".join(msg["content"] for msg in session.messages)
        prompt_tokens = estimate_tokens(prompt_text)
        session.prompt_tokens_processed += prompt_tokens

        # Generate deterministic response
        # Count user messages to determine turn number
        user_turn = sum(1 for m in session.messages if m["role"] == "user")

        if user_turn == 1:
            response = "Chào bạn! Mình là agent. Cần mình giúp gì không?"
        elif "tên" in message.lower() or "nhắc" in message.lower():
            response = "Mình không nhớ được tên vì mình chỉ nhớ trong cuộc hội thoại hiện tại."
        elif "ở đâu" in message.lower() or "nơi" in message.lower():
            response = "Mình không biết bạn ở đâu vì không lưu được thông tin dài hạn."
        elif "nghề" in message.lower() or "làm" in message.lower():
            response = "Mình không nhớ được nghề nghiệp của bạn."
        elif "thích" in message.lower() or "uống" in message.lower():
            response = "Mình không nhớ được preference của bạn."
        else:
            response = f"Đã nhận được: '{message[:50]}...'. Mình sẽ trả lời ngắn gọn."

        # Add assistant response
        session.messages.append({"role": "assistant", "content": response})

        # Estimate response tokens
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "response": response,
            "tokens_used": response_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire create_agent + InMemorySaver here."""
        # Live agent support - only when not in offline mode and model is available
        pass
