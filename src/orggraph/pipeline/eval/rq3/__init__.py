"""RQ3 pilot: R1 (vector-only) vs R2 (Cypher-by-qtype GraphRAG) on enron_qa_0922."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_CONDITIONS = ("R1", "R2")


@dataclass(frozen=True)
class Answer:
    """One retrieval-and-answer result, symmetric across R1 and R2."""

    text: str
    cited_email_ids: tuple[str, ...]
    retrieved_email_ids: tuple[str, ...]
    retrieved_bodies: tuple[str, ...]
    latency_ms: int
    condition: Literal["R1", "R2"]

    def __post_init__(self) -> None:
        if self.condition not in _VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {_VALID_CONDITIONS}, got {self.condition!r}"
            )
        if type(self.latency_ms) is not int:
            object.__setattr__(self, "latency_ms", int(self.latency_ms))


from orggraph.pipeline.eval.rq3.r1_vector import answer_r1  # noqa: E402
from orggraph.pipeline.eval.rq3.r2_graphrag import answer_r2  # noqa: E402

__all__ = ["Answer", "answer_r1", "answer_r2"]
