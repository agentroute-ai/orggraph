"""Thread-continuation experiment for RQ2.

A stronger evaluation design than judge-only naturalness scoring: take
a *real* email thread between two Enron employees, hold out the last
email as ground truth, and ask each architecture (multi-agent A2A vs
single-LLM long-context) to predict what that email should be. The
comparison is then anchored to actual human-written organisational
correspondence rather than to an LLM judge's intuition.

This module provides:

* :func:`load_thread` — pulls a 2-participant thread from
  ``clean_emails.parquet`` keyed by participant pair + normalised
  subject, sorted by timestamp.
* :func:`predict_next_multi_agent` — primes two
  :class:`~orggraph.agents.agent.PersonaAgent` instances with the
  prefix (each agent only sees the messages it sent or received) and
  asks the next-up speaker to produce one turn.
* :func:`predict_next_single_llm` — same prefix, but the single-LLM
  baseline sees the stitched personas + full conversation in one
  prompt and is asked to produce the next turn in the named speaker's
  voice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from orggraph.agents.agent import PersonaAgent, TextChatClient
from orggraph.agents.persona import Persona
from orggraph.simulation.single_llm import render_history, stitch_personas
from orggraph.simulation.transcript import Message, now_isoformat


def normalize_subject(s: object) -> str:
    """Strip ``Re:`` / ``Fw:`` / ``Fwd:`` prefixes and collapse whitespace."""
    if pd.isna(s):  # type: ignore[arg-type]
        return ""
    out = str(s).strip().lower()
    while True:
        new = re.sub(r"^(re|fw|fwd)\s*:\s*", "", out)
        if new == out:
            break
        out = new
    return re.sub(r"\s+", " ", out).strip()


def load_thread(
    parquet_path: Path | str,
    participants: tuple[str, str],
    subject: str,
) -> list[Message]:
    """Return a sorted list of :class:`Message` objects for one 2-participant thread.

    Parameters
    ----------
    parquet_path:
        Path to ``clean_emails.parquet``.
    participants:
        Two canonical sender names (order doesn't matter; the function
        accepts emails in either direction).
    subject:
        The normalised subject (output of :func:`normalize_subject`)
        the thread is keyed by.

    Returns
    -------
    Messages in chronological order, with body taken from
    ``body_truncated``. Each message's ``recipients`` is a single-name
    tuple — this loader only supports 2-participant single-recipient
    threads, which is the experimental scope of the comparison.
    """
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df["recipients_resolved"] = df["recipients_resolved"].apply(
        lambda r: list(r) if hasattr(r, "__iter__") and not isinstance(r, str) else []
    )
    df["recipient"] = df["recipients_resolved"].apply(lambda r: r[0] if r else None)
    df["subj_norm"] = df["subject"].apply(normalize_subject)

    pair = set(participants)
    mask = (
        df["sender_resolved"].isin(pair)
        & df["recipient"].isin(pair)
        & (df["sender_resolved"] != df["recipient"])
        & (df["subj_norm"] == subject)
    )
    thread = df[mask].sort_values("date").reset_index(drop=True)
    if thread.empty:
        return []

    out: list[Message] = []
    for i, row in thread.iterrows():
        body = str(row.get("body_truncated") or "").strip()
        ts = row.get("date")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        out.append(
            Message(
                sender=row["sender_resolved"],
                recipients=(row["recipient"],),
                body=body,
                turn_id=int(i),
                timestamp=ts_str,
                in_reply_to=int(i) - 1 if i > 0 else None,
            )
        )
    return out


def predict_next_multi_agent(
    prefix: list[Message],
    next_sender: str,
    next_recipient: str,
    personas: dict[str, Persona],
    client: TextChatClient,
    *,
    model: str,
    temperature: float = 0.7,
    scenario_brief: str = "Continue this real Enron email thread in the voice of the next sender.",
) -> Message:
    """Multi-agent (A2A) prediction.

    Sets up two :class:`PersonaAgent` instances, primes each with the
    prefix messages it would have seen on the Bus, and asks the
    ``next_sender`` agent to produce one reply addressed to
    ``next_recipient``.
    """
    if next_sender not in personas or next_recipient not in personas:
        raise KeyError(
            f"Both senders must be in personas; missing: "
            f"{[n for n in (next_sender, next_recipient) if n not in personas]}"
        )

    agents = {
        n: PersonaAgent(
            persona=personas[n], client=client, model=model, temperature=temperature,
        )
        for n in (next_sender, next_recipient)
    }
    # Per the A2A protocol: each agent sees only the messages it sent
    # or received. In a 2-person thread that means both agents see all
    # messages, but we go through is_visible_to() to keep the contract
    # consistent with the runner.
    for m in prefix:
        for agent in agents.values():
            if m.is_visible_to(agent.name):
                agent.receive(m)

    next_id = (max((m.turn_id for m in prefix), default=-1)) + 1
    return agents[next_sender].respond(
        scenario_brief=scenario_brief,
        recipients=(next_recipient,),
        turn_id=next_id,
        in_reply_to=next_id - 1 if prefix else None,
    )


def predict_next_single_llm(
    prefix: list[Message],
    next_sender: str,
    next_recipient: str,
    personas: dict[str, Persona],
    client: TextChatClient,
    *,
    model: str,
    temperature: float = 0.7,
    scenario_brief: str = "Continue this real Enron email thread in the voice of the next sender.",
) -> Message:
    """Single-LLM-with-long-context prediction.

    Stitches both personas into one system prompt, renders the full
    prefix as user-side context, and asks the model to produce the
    next message in ``next_sender``'s voice. Mirrors
    :func:`run_single_llm` but produces exactly one turn instead of
    looping.
    """
    if next_sender not in personas or next_recipient not in personas:
        raise KeyError(
            f"Both senders must be in personas; missing: "
            f"{[n for n in (next_sender, next_recipient) if n not in personas]}"
        )

    system_prompt = stitch_personas([personas[next_sender], personas[next_recipient]])
    user_content = (
        f"Scenario brief: {scenario_brief}\n\n"
        f"Conversation so far:\n{render_history(prefix)}\n\n"
        f"It is now the turn of: {next_sender}. "
        f"Write the next email — addressed to {next_recipient} — in {next_sender}'s "
        f"voice and style. Match how {next_sender} actually writes (vocabulary, "
        f"register, sign-off conventions). Speak directly; do not narrate. "
        f"Reply with only the email body — no quotation marks, no meta-commentary."
    )

    body = client.chat(
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        model=model,
        temperature=temperature,
    ).strip()

    next_id = (max((m.turn_id for m in prefix), default=-1)) + 1
    return Message(
        sender=next_sender,
        recipients=(next_recipient,),
        body=body,
        turn_id=next_id,
        timestamp=now_isoformat(),
        in_reply_to=next_id - 1 if prefix else None,
    )
