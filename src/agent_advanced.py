from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent User.md
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.sessions: dict[str, list[dict[str, str]]] = {}

        # Optionally initialize a real LangChain/LangGraph agent
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for this thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return prompt tokens processed for this thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return the size of the user's memory file in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions for this thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into User.md.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from User.md + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """
        # Step 1: Extract profile facts
        facts = extract_profile_updates(message)

        # Step 2: Persist facts to User.md
        current_profile = self.profile_store.read_text(user_id)
        profile_lines = []
        if current_profile.strip():
            profile_lines = [line.strip() for line in current_profile.split("\n") if line.strip()]

        # Build a map of existing keys
        existing_keys = {}
        new_lines = []
        for line in profile_lines:
            if ':' in line:
                # Extract key name
                key_part = line.split(':', 1)[0].strip().lstrip('-* ')
                # Find the key name
                for key in facts.keys():
                    if key.lower() == key_part.lower():
                        existing_keys[key.lower()] = len(new_lines)
                        break
                new_lines.append(line)
            else:
                new_lines.append(line)

        profile_lines = new_lines

        # Update or add facts
        for key, value in facts.items():
            key_lower = key.lower()
            if key_lower in existing_keys:
                # Update existing line
                idx = existing_keys[key_lower]
                profile_lines[idx] = f"- {key}: {value}"
            else:
                # Add new fact
                profile_lines.append(f"- {key}: {value}")

        # Write updated profile
        new_profile = "\n".join(profile_lines)
        if new_profile.strip():
            self.profile_store.write_text(user_id, new_profile)

        # Step 3: Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # Step 4: Estimate prompt context
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # Step 5: Generate response
        response = self._offline_response(user_id, thread_id, message)

        # Step 6: Append assistant reply and update counters
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(response)
        )

        # Also track in session for recall lookup
        if thread_id not in self.sessions:
            self.sessions[thread_id] = []
        self.sessions[thread_id].append({"role": "user", "content": message})
        self.sessions[thread_id].append({"role": "assistant", "content": response})

        return {
            "response": response,
            "tokens_used": estimate_tokens(response),
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Include User.md, compact summary text, and recent kept messages.
        """
        tokens = 0

        # User.md content
        user_profile = self.profile_store.read_text(user_id)
        tokens += estimate_tokens(user_profile)

        # Compact summary
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        tokens += estimate_tokens(summary)

        # Recent kept messages
        messages = ctx.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            tokens += estimate_tokens(content)

        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Can answer questions about:
        - Name
        - Location
        - Profession
        - Preferences
        - Response style
        """
        profile = self.profile_store.read_text(user_id)
        facts = {}
        for line in profile.split("\n"):
            line = line.strip().lstrip("-* ").strip()
            if ": " in line or ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    facts[key] = value

        # Parse facts into structured data
        name = facts.get("name", "")
        location = facts.get("location", "")
        profession = facts.get("profession", "")
        favorite_drink = facts.get("favorite_drink", "")
        favorite_food = facts.get("favorite_food", "")
        response_style = facts.get("response_style", "")
        pet = facts.get("pet", "")

        msg_lower = message.lower()

        # Answer recall questions
        if "tên" in msg_lower or ("nhắc" in msg_lower and "tên" in msg_lower):
            if name:
                parts = [f"Tên bạn là {name}."]
                if favorite_drink:
                    parts.append(f"Đồ uống yêu thích: {favorite_drink}.")
                return " ".join(parts)
            return "Mình chưa nhớ được tên của bạn."

        if "đồ uống" in msg_lower or "yêu thích" in msg_lower and "uống" in msg_lower:
            if favorite_drink:
                return f"Đồ uống yêu thích của bạn là {favorite_drink}."
            return "Mình chưa nhớ được đồ uống yêu thích."

        if "món ăn" in msg_lower or ("yêu thích" in msg_lower and "ăn" in msg_lower):
            if favorite_food:
                return f"Món ăn yêu thích của bạn là {favorite_food}."
            return "Mình chưa nhớ được món ăn yêu thích."

        if "ở đâu" in msg_lower or "nơi" in msg_lower or "tp" in msg_lower or "thành phố" in msg_lower:
            if location:
                return f"Hiện bạn đang ở {location}."
            return "Mình chưa nhớ được nơi bạn ở."

        if "nghề" in msg_lower or "làm" in msg_lower or "công việc" in msg_lower:
            if profession:
                return f"Hiện bạn đang làm {profession}."
            return "Mình chưa nhớ được nghề nghiệp của bạn."

        if "style" in msg_lower or "trả lời" in msg_lower or "thích" in msg_lower:
            if response_style:
                return f"Style trả lời bạn thích: {response_style}."
            return "Mình chưa nhớ được style trả lời bạn thích."

        if "con gì" in msg_lower or "nuôi" in msg_lower or "pet" in msg_lower:
            if pet:
                return f"Bạn nuôi {pet}."
            return "Mình chưa nhớ được bạn nuôi con gì."

        if "tóm tắt" in msg_lower or "biết" in msg_lower and ("ai" in msg_lower or "gì" in msg_lower):
            parts = []
            if name:
                parts.append(f"Tên: {name}")
            if profession:
                parts.append(f"Nghề: {profession}")
            if location:
                parts.append(f"Địa điểm: {location}")
            if favorite_drink:
                parts.append(f"Đồ uống: {favorite_drink}")
            if favorite_food:
                parts.append(f"Món ăn: {favorite_food}")
            if parts:
                return " ".join(parts)
            return "Mình đang thu thập thông tin về bạn."

        if "nhớ" in msg_lower or "ghi nhớ" in msg_lower:
            return "OK, mình đã ghi nhận. Bạn cứ hỏi lại khi cần nhé!"

        if "chào" in msg_lower or "xin chào" in msg_lower:
            parts = ["Chào bạn!"]
            if name:
                parts.append(f"Mình đã nhớ bạn là {name}.")
            return " ".join(parts)

        # Default response
        return "Đã nhận được. Mình sẽ nhớ thông tin này cho lần sau."

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware.

        High-level design:
        - build_chat_model(self.config.model) for the selected provider
        - InMemorySaver for short-term thread state
        - tool to read User.md
        - tool to write/edit User.md
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """
        # Live agent support - only when not in offline mode
        pass
