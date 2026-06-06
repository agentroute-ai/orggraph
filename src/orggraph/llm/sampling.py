"""Email sampling for LLM prompts.

Three sampling strategies — one per prompt type — derived from the
deferred Tier 1/Tier 2 spec and extended for entity-level extraction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_MAX_BODY = 1200
DEFAULT_MAX_SUBJECT = 120


def _row_to_dict(row: pd.Series, max_body: int, max_subject: int) -> dict[str, str]:
    return {
        "from": str(row.get("from", ""))[:200],
        "to": str(row.get("to", ""))[:500],
        "subject": str(row.get("subject", ""))[:max_subject],
        "body": str(row.get("body", ""))[:max_body],
        "date": str(row.get("date", "")),
    }


def sample_emails_for_person(
    emails: pd.DataFrame,
    person: str,
    max_body_chars: int = DEFAULT_MAX_BODY,
    max_subject_chars: int = DEFAULT_MAX_SUBJECT,
) -> list[dict[str, str]]:
    """Sample 20 representative emails for a Person.

    Mix per the deferred spec:
      - 8 longest outbound (person is sender)
      - 8 longest inbound (person is in recipients_resolved)
      - 4 thread-initiation (outbound, subject not Re:/Fw:)
    Deduplicated; truncated to body/subject limits.
    """
    if "sender_resolved" not in emails.columns:
        return []

    if "body_len" not in emails.columns:
        emails = emails.assign(body_len=emails["body"].astype(str).str.len())

    outbound = emails[emails["sender_resolved"] == person]
    if outbound.empty and not emails["recipients_resolved"].apply(lambda r: person in r if isinstance(r, list) else False).any():
        return []

    out_top = outbound.nlargest(8, "body_len")
    inbound = emails[
        emails["recipients_resolved"].apply(
            lambda r: person in r if isinstance(r, list) else False
        )
    ]
    in_top = inbound.nlargest(8, "body_len")

    # Thread starts (subject doesn't begin with Re:/Fw:/Fwd:)
    if not outbound.empty:
        subj = outbound["subject"].astype(str).str.lower()
        starts = outbound[~subj.str.startswith(("re:", "fw:", "fwd:"))]
        thread_starts = starts.nlargest(4, "body_len")
    else:
        thread_starts = outbound.head(0)

    pick = pd.concat([out_top, in_top, thread_starts])
    pick = pick.drop_duplicates(subset=["from", "subject", "body", "date"]).head(20)

    return [_row_to_dict(r, max_body_chars, max_subject_chars) for _, r in pick.iterrows()]


def sample_emails_for_entity(
    emails: pd.DataFrame,
    domain: str,
    k: int = 10,
    max_body_chars: int = DEFAULT_MAX_BODY,
    max_subject_chars: int = DEFAULT_MAX_SUBJECT,
) -> list[dict[str, str]]:
    """Sample top-K longest email exchanges with the given external domain.

    An exchange counts if the domain appears in either `from` or `to`.
    """
    domain_lower = domain.lower()
    mask = (
        emails["from"].astype(str).str.lower().str.contains(domain_lower, regex=False, na=False)
        | emails["to"].astype(str).str.lower().str.contains(domain_lower, regex=False, na=False)
    )
    matched = emails[mask]
    if matched.empty:
        return []

    if "body_len" not in matched.columns:
        matched = matched.assign(body_len=matched["body"].astype(str).str.len())

    top = matched.nlargest(k, "body_len")
    return [_row_to_dict(r, max_body_chars, max_subject_chars) for _, r in top.iterrows()]


def sample_emails_for_pair(
    emails: pd.DataFrame,
    person_a: str,
    person_b: str,
    k: int = 5,
    min_exchanges: int = 5,
    max_body_chars: int = DEFAULT_MAX_BODY,
    max_subject_chars: int = DEFAULT_MAX_SUBJECT,
) -> list[dict[str, Any]]:
    """Sample up to K emails exchanged between two persons.

    Returns [] if total exchanges < min_exchanges. Each returned item has a
    `sender` key in addition to the standard fields.
    """
    if "sender_resolved" not in emails.columns or "recipients_resolved" not in emails.columns:
        return []

    # Vectorized: filter by sender first (cheap mask), then check recipients on the smaller frame
    sender_a_mask = emails["sender_resolved"] == person_a
    sender_b_mask = emails["sender_resolved"] == person_b

    def _has_recip(recips, target):
        return target in recips if isinstance(recips, list) else False

    a_to_b = emails[sender_a_mask & emails["recipients_resolved"].apply(
        lambda r: _has_recip(r, person_b)
    )]
    b_to_a = emails[sender_b_mask & emails["recipients_resolved"].apply(
        lambda r: _has_recip(r, person_a)
    )]

    total = len(a_to_b) + len(b_to_a)
    if total < min_exchanges:
        return []

    if "body_len" not in a_to_b.columns:
        a_to_b = a_to_b.assign(body_len=a_to_b["body"].astype(str).str.len())
    if "body_len" not in b_to_a.columns:
        b_to_a = b_to_a.assign(body_len=b_to_a["body"].astype(str).str.len())

    half_k = max(1, k // 2)
    a_top = a_to_b.nlargest(half_k, "body_len")
    b_top = b_to_a.nlargest(k - half_k, "body_len")

    out: list[dict[str, Any]] = []
    for _, r in a_top.iterrows():
        d = _row_to_dict(r, max_body_chars, max_subject_chars)
        d["sender"] = person_a
        out.append(d)
    for _, r in b_top.iterrows():
        d = _row_to_dict(r, max_body_chars, max_subject_chars)
        d["sender"] = person_b
        out.append(d)
    return out[:k]
