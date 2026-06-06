"""LLM judge for KG-Chat eval questions whose grading kind is 'judge'.

The judge prompt is supplied per-question in the YAML
(grading.prompt). The judge sees the prompt + the agent's answer and
returns a JSON verdict.
"""

from __future__ import annotations
import json
import re


JUDGE_PROMPT_TEMPLATE = """You are evaluating an assistant's answer to a question about the Enron email corpus.

Question:
{question}

Assistant's answer:
{answer}

Evaluation criterion:
{criterion}

Respond with one JSON object on a single line: {{"verdict": "pass" | "fail", "reason": "<one short sentence>"}}.
"""

_JSON_FENCE = re.compile(r"\{[^{}]*\}")


def judge_answer(
    *,
    client,           # OpenAIChatClient
    model: str,
    question: str,
    answer: str,
    criterion: str,
    temperature: float = 0.0,
) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer, criterion=criterion)
    raw = client.chat(
        system="You are a strict evaluator. Reply with one JSON object only.",
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> tuple[bool, str]:
    if not raw:
        return False, "empty judge response"
    raw = raw.strip()
    candidates: list[str] = []
    if raw.startswith("{"):
        candidates.append(raw)
    candidates.extend(_JSON_FENCE.findall(raw))
    for blob in candidates:
        try:
            obj = json.loads(blob)
            verdict = str(obj.get("verdict", "")).strip().lower() == "pass"
            return verdict, str(obj.get("reason", "")).strip()
        except json.JSONDecodeError:
            continue
    return False, "unparseable judge response"
