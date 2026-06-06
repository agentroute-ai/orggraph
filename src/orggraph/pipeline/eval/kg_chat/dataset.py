"""Loader for the KG-Chat structural eval question set.

The questions live as a static YAML at the same package location.
Each row carries a question, expected_tools annotation, and a grading
clause that ``grader.py`` understands.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    expected_tools: tuple[str, ...]
    grading: dict                              # raw grading clause (kind + parameters)
    team_grading: dict = field(default_factory=dict)
    """Optional routing/cascade GT for team-eval questions. Empty dict
    for hard-eval / single-agent questions so callers that don't pass
    it (e.g. route() constructing a synthetic Question) keep working."""


def load_questions(path: Path | None = None) -> list[Question]:
    """Read questions.yaml and return a list of Question objects.

    ``team_grading`` (optional) captures routing + cascade ground truth
    for team-chat eval questions:

      team_grading:
        expected_initial: [slug_a, slug_b, ...]
        expected_targets:
          category_name: [slug_x, slug_y, ...]
        min_categories_covered: 2
    """
    path = path or DEFAULT_QUESTIONS_PATH
    data = yaml.safe_load(path.read_text())
    out: list[Question] = []
    for q in data.get("questions", []):
        out.append(Question(
            id=q["id"],
            category=q["category"],
            question=q["question"],
            expected_tools=tuple(q.get("expected_tools") or []),
            grading=dict(q.get("grading") or {}),
            team_grading=dict(q.get("team_grading") or {}),
        ))
    return out
