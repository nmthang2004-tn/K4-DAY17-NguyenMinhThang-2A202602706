from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear.

    - 0: no expected fact found
    - 0.5: some expected facts found
    - 1: all expected facts found
    """
    if not expected:
        return 1.0

    answer_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in answer_lower)

    if matches == 0:
        return 0.0
    elif matches == len(expected):
        return 1.0
    else:
        return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode.

    Measures:
    - Completeness: how many expected facts are present
    - Conciseness: is the answer reasonably short
    - Specificity: does the answer use specific values vs vague responses
    """
    if not answer:
        return 0.0

    score = 0.0

    # Completeness (0-0.5)
    matches = sum(1 for exp in expected if exp.lower() in answer.lower())
    if expected:
        completeness = matches / len(expected)
        score += completeness * 0.5

    # Conciseness (0-0.3) - penalize very long or very short answers
    word_count = len(answer.split())
    if 5 <= word_count <= 30:
        score += 0.3
    elif 3 <= word_count < 5 or 30 < word_count <= 50:
        score += 0.15

    # Specificity (0-0.2) - reward using specific values
    has_specific_info = any(char.isdigit() for char in answer) or any(
        term in answer for term in ["là", "ở", "-", ":"]
    )
    if has_specific_info:
        score += 0.2

    return min(score, 1.0)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track agent tokens only.
    3. Track prompt tokens processed.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_questions = 0

    # Track memory growth
    initial_memory_sizes: dict[str, int] = {}
    final_memory_sizes: dict[str, int] = {}

    total_compactions = 0

    for conv in conversations:
        user_id = conv.get("user_id", "unknown")
        turns = conv.get("turns", [])
        recall_questions = conv.get("recall_questions", [])

        # Record initial memory size for advanced agent
        if agent_name == "Advanced" and hasattr(agent, "profile_store"):
            initial_memory_sizes[user_id] = agent.profile_store.file_size(user_id)

        # Feed all turns
        thread_id = f"conv-{conv.get('id', 'unknown')}"

        for turn in turns:
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("tokens_used", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Ask recall questions in a fresh thread (cross-session test)
        recall_thread_id = f"recall-{conv.get('id', 'unknown')}"

        for rq in recall_questions:
            question = rq.get("question", "")
            expected = rq.get("expected_contains", [])

            # Get response from agent
            result = agent.reply(user_id, recall_thread_id, question)
            answer = result.get("response", "")

            # Calculate recall and quality
            recall = recall_points(answer, expected)
            quality = heuristic_quality(answer, expected)

            total_recall += recall
            total_quality += quality
            total_questions += 1

        # Track compactions
        total_compactions += agent.compaction_count(thread_id)

        # Record final memory size for advanced agent
        if agent_name == "Advanced" and hasattr(agent, "profile_store"):
            final_memory_sizes[user_id] = agent.profile_store.file_size(user_id)

    # Calculate memory growth
    memory_growth = 0
    if agent_name == "Advanced":
        for user_id in final_memory_sizes:
            initial = initial_memory_sizes.get(user_id, 0)
            final = final_memory_sizes[user_id]
            memory_growth += final - initial

    # Calculate averages
    avg_recall = total_recall / total_questions if total_questions > 0 else 0.0
    avg_quality = total_quality / total_questions if total_questions > 0 else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    if not rows:
        return "No results to display."

    headers = [
        "Agent",
        "Agent Tokens",
        "Prompt Tokens",
        "Recall",
        "Quality",
        "Memory Growth (bytes)",
        "Compactions",
    ]

    table_data = []
    for row in rows:
        table_data.append([
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            row.recall_score,
            row.response_quality,
            row.memory_growth_bytes,
            row.compactions,
        ])

    return tabulate(table_data, headers=headers, tablefmt="grid")


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """
    config = load_config(Path(__file__).resolve().parent.parent)

    # Load both datasets
    standard_data = load_conversations(config.data_dir / "conversations.json")
    stress_data = load_conversations(config.data_dir / "advanced_long_context.json")

    # Initialize agents
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    print("=" * 80)
    print("Standard Benchmark")
    print("=" * 80)

    # Run standard benchmark
    baseline_std = run_agent_benchmark("Baseline", baseline, standard_data, config)
    advanced_std = run_agent_benchmark("Advanced", advanced, standard_data, config)

    print(format_rows([baseline_std, advanced_std]))
    print()

    print("=" * 80)
    print("Long-Context Stress Benchmark")
    print("=" * 80)

    # Reset agents for stress test
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    baseline_stress_row = run_agent_benchmark("Baseline", baseline_stress, stress_data, config)
    advanced_stress_row = run_agent_benchmark("Advanced", advanced_stress, stress_data, config)

    print(format_rows([baseline_stress_row, advanced_stress_row]))


if __name__ == "__main__":
    main()
