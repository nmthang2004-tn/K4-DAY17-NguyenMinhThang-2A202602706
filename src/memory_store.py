from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Simple token estimator.

    Strip whitespace, return 0 for empty text,
    approximate tokens from character count (len / 4).
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # Rough estimate: ~4 characters per token on average
    return len(stripped) // 4 + 1


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Map each user id to one markdown file.
    Support read / write / edit operations.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        """Return the path for a user's profile markdown file."""
        # Slugify user_id to create a safe filename
        safe_name = re.sub(r'[^\w\-.]', '_', user_id)
        return self.root_dir / f"{safe_name}.md"

    def read_text(self, user_id: str) -> str:
        """Return file content or an empty default markdown profile."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md and return whether it changed."""
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0


# Regex patterns for extracting profile facts from Vietnamese text
# Combined pattern for extracting facts - simpler approach
_NAME_PATTERNS = [
    r'(?:mình|tôi|tao)\s+tên\s+là\s+([A-Za-zÀ-ỹ]+)',
    r'tên\s+(?:là|là\s+)?([A-Za-zÀ-ỹ]+?)(?=[,\s]|$)',
]

_LOCATION_PATTERNS = [
    r'(?:đang|đã)\s+(?:ở|làm)\s+([A-Za-zÀ-ỹ\s]+?)(?=[,\s]|$)',
    r'ở\s+([A-Za-zÀ-ỹ\s]+?)(?=[,\s]|$)',
]

_PROFESSION_PATTERNS = [
    r'(?:đang làm|là)\s+((?:backend|frontend|fullstack|mlops|devops|data|product|design)\s*engineer)',
    r'(?:backend|frontend|fullstack|mlops|devops|data)\s*engineer',
    r'chuyển\s+sang\s+([A-Za-zÀ-ỹ\s]+?)\s*engineer',
]

_PREFERENCE_PATTERNS = [
    r'(?:đồ uống|yêu thích|uống)\s+(?:là\s+)?(cà phê sữa đá|cà phê|trà|sữa)',
    r'cà phê sữa đá',
    r'món\s+ăn\s+yêu\s+thích\s+(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?=[,.\s]|$)',
]

_STYLE_PATTERNS = [
    r'(?:trả lời)\s+((?:ngắn gọn|thành\s+bullet|3\s+bullet)[A-Za-zÀ-ỹ\s,]*)',
]

_HOBBY_PATTERNS = []

_PET_PATTERNS = [
    r'nuôi\s+([A-Za-zÀ-ỹ\s]+?)\s+tên\s+([A-Za-zÀ-ỹ]+)',
]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Extracts:
    - name
    - location
    - profession
    - preferences (drink, food)
    - response style
    - pets

    Skips questions (messages ending with ? or containing question words).
    """
    facts: dict[str, str] = {}
    msg_lower = message.lower()
    msg_stripped = message.strip()

    # Skip questions - don't extract facts from questions
    # Questions typically end with "?" or contain question words
    if msg_stripped.endswith('?') or 'gì?' in msg_stripped or 'không?' in msg_stripped:
        return facts

    # Skip if it's clearly a question with interrogative patterns
    question_patterns = [
        r'^tên\s+mình',  # "Tên mình là gì?"
        r'^bạn\s+(?:có\s+)?thể',  # "Bạn có thể..."
        r'^mình\s+muốn\s+hỏi',  # "Mình muốn hỏi..."
        r'^nhờ\s+bạn',  # "Nhờ bạn..."
        r'nhắc\s+lại',  # "Nhắc lại giúp mình..."
        r'^bạn\s+thử',  # "Bạn thử..."
    ]
    for pattern in question_patterns:
        if re.search(pattern, msg_stripped, re.IGNORECASE):
            return facts

    # Extract name - look for "tên là X" or "mình tên là X"
    # Handle multi-word names like "DũngCT Stress"
    name_patterns = [
        r'mình\s+tên\s+là\s+([A-Za-zÀ-ỹ]+(?:[\s]+[A-Za-zÀ-ỹ]+)?)',
        r'tên\s+là\s+([A-Za-zÀ-ỹ]+(?:[\s]+[A-Za-zÀ-ỹ]+)?)',
        r'tôi\s+tên\s+là\s+([A-Za-zÀ-ỹ]+(?:[\s]+[A-Za-zÀ-ỹ]+)?)',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove trailing punctuation
            name = re.sub(r'[.,;:]+$', '', name)
            if len(name) > 1 and len(name) < 50:
                facts["name"] = name
                break

    # Extract location - check for corrections first (more recent = more important)
    # Use greedy match with specific ending patterns
    location_patterns = [
        r'chuyển\s+sang\s+([A-Za-zÀ-ỹ\s]{2,30}?)(?=\s*(?:rồi|làm|$))',
        r'đang\s+ở\s+([A-Za-zÀ-ỹ\s]{2,40}?)(?=\s+làm|\s+working|\s+ở|,)',
        r'ở\s+([A-Za-zÀ-ỹ\s]{2,40}?)(?=\s+làm|\s+working|,\s|$)',
    ]
    for pattern in location_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            loc = match.group(1).strip()
            # Clean up common noise
            loc = re.sub(r'^(tp|thành phố|thành)\s+', '', loc, flags=re.IGNORECASE)
            if len(loc) > 2 and len(loc) < 30 and loc.lower() not in ['đây', 'đó', 'đây là', 'này', 'làm việc', 'làm']:
                facts["location"] = loc
                break
    # Fallback: try simpler pattern
    if "location" not in facts:
        simple_match = re.search(r'ở\s+([A-Za-zÀ-ỹ\s]+?)(?=[,\s]|$)', message)
        if simple_match:
            loc = simple_match.group(1).strip()
            if len(loc) > 2 and len(loc) < 30:
                facts["location"] = loc

    # Extract profession
    profession_patterns = [
        r'(?:đang\s+)?làm\s+((?:backend|frontend|fullstack|mlops|devops|data)\s*engineer)',
        r'((?:backend|frontend|fullstack|mlops|devops|data)\s*engineer)',
        r'chuyển\s+sang\s+([A-Za-zÀ-ỹ\s]+?)\s*engineer',
    ]
    for pattern in profession_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            facts["profession"] = match.group(1).strip()
            break

    # Extract drink preference
    if 'cà phê sữa đá' in msg_lower or 'café sữa đá' in msg_lower:
        facts["favorite_drink"] = "cà phê sữa đá"
    elif 'cà phê' in msg_lower:
        drink_match = re.search(r'(?:uống\s+)?cà phê\s*([A-Za-zÀ-ỹ\s]*?)(?=[,.\s]|$)', message, re.IGNORECASE)
        if drink_match:
            drink = drink_match.group(1).strip() or 'cà phê'
            if drink:
                facts["favorite_drink"] = drink if len(drink) < 20 else "cà phê"

    # Extract food preference
    food_match = re.search(r'món\s+ăn\s+yêu\s+thích\s+(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?=[,.\s]|$)', message, re.IGNORECASE)
    if food_match:
        food = food_match.group(1).strip()
        if len(food) > 1:
            facts["favorite_food"] = food

    # Extract response style
    style_parts = []
    if 'ngắn gọn' in msg_lower:
        style_parts.append("ngắn gọn")
    if 'bullet' in msg_lower:
        style_parts.append("3 bullet")
    if 'ví dụ' in msg_lower or 'thực chiến' in msg_lower:
        style_parts.append("có ví dụ thực chiến")
    if 'trade-off' in msg_lower or 'so sánh' in msg_lower:
        style_parts.append("nhấn trade-off")
    if style_parts:
        facts["response_style"] = ", ".join(style_parts)

    # Extract pet info
    pet_match = re.search(r'nuôi\s+([A-Za-zÀ-ỹ\s]+?)\s+tên\s+([A-Za-zÀ-ỹ]+)', message, re.IGNORECASE)
    if pet_match:
        pet_type = pet_match.group(1).strip()
        pet_name = pet_match.group(2).strip()
        if pet_type and pet_name:
            facts["pet"] = f"{pet_type} tên {pet_name}"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    Heuristic text concatenation that keeps key facts.
    """
    if not messages:
        return ""

    # Keep the most recent messages as context
    recent = messages[-max_items:] if len(messages) > max_items else messages

    summary_parts = []

    # Extract key information from recent messages
    for msg in recent:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        if role == "user":
            # Try to extract profile facts from user messages
            facts = extract_profile_updates(content)
            for key, value in facts.items():
                summary_parts.append(f"- {key}: {value}")
        elif role == "assistant" and content:
            # Keep short summaries of assistant responses
            if len(content) < 100:
                summary_parts.append(f"[Assistant]: {content[:80]}...")

    # If we have too many parts, condense further
    if len(summary_parts) > 10:
        return f"[Summary of {len(messages)} messages]: " + "; ".join(summary_parts[:5]) + "..."

    return "\n".join(summary_parts) if summary_parts else ""


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keep recent messages in full.
    When the thread grows too large, move older content into a summary.
    Track how many compactions happened for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread and trigger compaction if needed."""
        # Create thread state if missing
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        # Append the new message
        thread_state = self.state[thread_id]
        messages = thread_state["messages"]
        messages.append({"role": role, "content": content})

        # Check if compaction is needed
        total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
        if total_tokens > self.threshold_tokens:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> None:
        """Compact messages older than keep_messages."""
        thread_state = self.state[thread_id]
        messages = thread_state["messages"]

        if len(messages) <= self.keep_messages:
            return

        # Separate recent messages from older ones
        keep = messages[-self.keep_messages:]
        older = messages[:-self.keep_messages]

        # Create summary of older messages
        summary = summarize_messages(older, max_items=6)
        thread_state["summary"] = summary
        thread_state["messages"] = keep
        thread_state["compactions"] = thread_state.get("compactions", 0) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with messages, summary, compactions."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        if thread_id not in self.state:
            return 0
        return self.state[thread_id].get("compactions", 0)
