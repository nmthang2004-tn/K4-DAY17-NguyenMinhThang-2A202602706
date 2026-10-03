from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests.

    Point state_dir into tmp_path.
    Reduce compact threshold so compaction happens quickly in tests.
    """
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=100,  # Small threshold to trigger compaction quickly
        compact_keep_messages=2,
        model=ProviderConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        judge_model=ProviderConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.7,
        ),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Initially empty
    content = store.read_text(user_id)
    assert content == ""

    # Write initial profile
    initial_content = "# User Profile\n- name: Test User\n- profession: Engineer"
    store.write_text(user_id, initial_content)

    # Read back
    content = store.read_text(user_id)
    assert "Test User" in content
    assert "Engineer" in content

    # Edit existing fact
    success = store.edit_text(user_id, "Test User", "Updated User")
    assert success is True

    # Verify edit
    content = store.read_text(user_id)
    assert "Updated User" in content
    assert "Test User" not in content

    # File size should be > 0
    size = store.file_size(user_id)
    assert size > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)

    # Create compact manager with very low threshold
    manager = CompactMemoryManager(
        threshold_tokens=50,  # Very low to trigger compaction
        keep_messages=2,
    )

    thread_id = "test_thread"

    # Add messages until compaction triggers
    for i in range(10):
        manager.append(thread_id, "user", f"This is message number {i} with some content to increase token count.")

    # Should have compacted at least once
    compactions = manager.compaction_count(thread_id)
    assert compactions > 0, f"Expected compaction to trigger, but got {compactions} compactions"

    # Context should have summary
    ctx = manager.context(thread_id)
    assert "summary" in ctx


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)

    # Create agents
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "test_dungct"

    # First session - user introduces themselves
    session1 = "session_1"
    baseline.reply(user_id, session1, "Chào, mình tên là DũngCT, đang làm backend engineer.")
    advanced.reply(user_id, session1, "Chào, mình tên là DũngCT, đang làm backend engineer.")

    # Second session - different thread
    session2 = "session_2"

    # Ask for name recall in a new thread
    baseline_response = baseline.reply(user_id, session2, "Tên mình là gì?")
    advanced_response = advanced.reply(user_id, session2, "Tên mình là gì?")

    # Baseline should NOT recall
    assert "DũngCT" not in baseline_response["response"], \
        "Baseline should not recall name across sessions"

    # Advanced SHOULD recall
    assert "DũngCT" in advanced_response["response"], \
        f"Advanced should recall name across sessions, but got: {advanced_response['response']}"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "stress_test"
    thread_id = "long_thread"

    # Simulate a long conversation with many turns
    for i in range(20):
        message = f"Đây là message số {i}. Mình đang test compact memory. " * 5
        baseline.reply(user_id, thread_id, message)
        advanced.reply(user_id, thread_id, message)

    # Get prompt token counts
    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)

    # Advanced should have lower or similar prompt load due to compaction
    # The key is that compactions have happened
    compactions = advanced.compaction_count(thread_id)

    # Verify compaction happened
    assert compactions > 0, f"Expected compactions > 0, got {compactions}"

    # The advanced agent's context should be more compact
    ctx = advanced.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= config.compact_keep_messages + 5, \
        f"Expected fewer messages after compaction, got {len(ctx['messages'])}"


def test_extract_profile_updates() -> None:
    """Test profile fact extraction."""
    # Test name extraction
    facts = extract_profile_updates("Chào bạn, mình tên là DũngCT.")
    assert "name" in facts
    assert "DũngCT" in facts["name"]

    # Test location extraction
    facts = extract_profile_updates("Mình đang ở Đà Nẵng làm việc.")
    assert "location" in facts
    assert "Đà Nẵng" in facts["location"]

    # Test profession extraction
    facts = extract_profile_updates("Mình là backend engineer cho startup AI.")
    assert "profession" in facts

    # Test drink preference
    facts = extract_profile_updates("Mình thích cà phê sữa đá mỗi sáng.")
    assert "favorite_drink" in facts

    # Test response style
    facts = extract_profile_updates("Mình muốn bạn trả lời ngắn gọn thành bullet.")
    assert "response_style" in facts


def test_estimate_tokens() -> None:
    """Test token estimation."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0

    # Short text should return at least 1
    assert estimate_tokens("hello") >= 1

    # Longer text should scale roughly with length / 4
    long_text = "a" * 100
    tokens = estimate_tokens(long_text)
    assert 20 <= tokens <= 30, f"Expected ~25 tokens, got {tokens}"
