"""Few-shot writing-sample loader for agent grounding.

Pulls a small stratified set of sender-only emails from
``clean_emails.parquet`` and formats them as drop-in
:attr:`Persona.sample_messages` blocks. The samples give the model
concrete style references (vocabulary, sign-off, sentence rhythm) on
top of the abstract behavioural profile, which we observed to reduce
the speaker-tracking failure mode in the single-LLM long-context
condition (the model addressing itself by name in turn 1).

Stratification picks samples by length quantile so the model sees both
short, terse messages and longer, more deliberate ones. This is a
crude proxy for the speech-act stratification used in Stage 4a; if we
need richer stratification for the full pilot we can swap it for a
query that reads ``embeddings_email`` joined with the ``speech_acts``
JSONB column.

For *thread continuation* experiments — where we predict an email that
was actually written on date X — pass ``before_date=X`` so the
selector cannot leak post-X writing into the agent prompt. This is the
canonical way to enforce temporal hygiene in the experimental design;
without it, the agent's "stylistic grounding" can carry hindsight
information from later in the corpus into a prediction about an
earlier moment.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path("datasets/enron/processed/clean_emails.parquet")


# Forwarded / quoted-content markers. Bodies that begin with one of these
# are mostly someone else's text — the model would learn to fabricate
# invented forwards rather than imitating the persona's own voice. We
# observed this regression on the Tracy/Rod thread when Rod's pair-
# specific samples included real forwarded emails: the model copied
# the template and made up content (e.g. an invented "$50M liquidity
# from Kathy Brown" email). The filter is conservative — it only matches
# canonical Outlook / Lotus Notes forwarding headers near the top of
# the body, not legitimate uses of "Forwarded" mid-text.
_FORWARDED_MARKERS = (
    "----- Forwarded by",
    "-----Forwarded by",
    "----- Original Message",
    "-----Original Message",
    "From:",  # only at start-of-body, after stripping
)


def _looks_forwarded(body: str) -> bool:
    """Return True if ``body`` looks dominated by forwarded / quoted content.

    We check the first ~200 characters: if any canonical forwarding
    header appears there, the body's substance is someone else's
    writing rather than this person's. The "From:" prefix only
    triggers at the very start, since legitimate emails often
    contain "From:" elsewhere.
    """
    head = body.strip()[:200]
    for marker in _FORWARDED_MARKERS:
        if marker == "From:":
            if head.startswith("From:"):
                return True
        elif marker in head:
            return True
    return False


def load_pair_samples(
    sender: str,
    recipient: str,
    *,
    parquet_path: Path | None = None,
    n_samples: int = 3,
    max_body_chars: int = 240,
    min_body_chars: int = 80,
    before_date: dt.datetime | str | pd.Timestamp | None = None,
    skip_forwarded: bool = True,
) -> tuple[str, ...]:
    """Like :func:`load_sample_messages` but restricted to ``sender → recipient``.

    Real organisational writing varies sharply by audience: the same
    person speaks differently to their boss, their direct reports, and
    a peer in another team. Pair-specific samples expose this
    relationship-conditional register to the agent — useful for both
    the dialogue pilot and the thread-continuation experiment, where
    the prediction target is always addressed to a specific recipient.

    Same temporal-hygiene contract as :func:`load_sample_messages`:
    ``before_date`` enforces strict <, samples at-or-after the cutoff
    are excluded.

    Returns an empty tuple if the (sender, recipient) ordered pair has
    no qualifying emails (very common; the corpus is sparse).
    """
    p = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    if not p.exists():
        return ()

    cols = [
        "sender_resolved", "recipients_resolved",
        "subject", "body_truncated", "body_chars",
    ]
    if before_date is not None:
        cols.append("date")
    df = pd.read_parquet(p, columns=cols)

    df = df.copy()
    df["recipients_resolved"] = df["recipients_resolved"].apply(
        lambda r: list(r) if hasattr(r, "__iter__") and not isinstance(r, str) else []
    )
    sent = df[
        (df["sender_resolved"] == sender)
        & df["recipients_resolved"].apply(lambda r: recipient in r)
    ].copy()
    if sent.empty:
        return ()

    if before_date is not None:
        cutoff = pd.to_datetime(before_date, utc=True)
        sent_dates = pd.to_datetime(sent["date"], utc=True, errors="coerce")
        sent = sent[sent_dates < cutoff]
        if sent.empty:
            return ()

    sent = sent[sent["body_chars"] >= min_body_chars]
    if sent.empty:
        return ()

    if skip_forwarded:
        mask = sent["body_truncated"].fillna("").apply(
            lambda b: not _looks_forwarded(str(b))
        )
        sent = sent[mask]
        if sent.empty:
            return ()

    sent = sent.sort_values("body_chars").reset_index(drop=True)

    if len(sent) <= n_samples:
        idx = list(range(len(sent)))
    else:
        idx = [
            int(round((q / (n_samples + 1)) * (len(sent) - 1)))
            for q in range(1, n_samples + 1)
        ]

    out: list[str] = []
    for i in idx:
        row = sent.iloc[i]
        body = str(row.get("body_truncated") or "").strip()
        if not body:
            continue
        if len(body) > max_body_chars:
            body = body[: max_body_chars - 1].rstrip() + "…"
        body = body.replace("\n", "\n  ")
        subject = str(row.get("subject") or "").strip() or "(no subject)"
        out.append(f"To: {recipient}\n  Subject: {subject}\n  {body}")
    return tuple(out)


def load_sample_messages(
    name: str,
    *,
    parquet_path: Path | None = None,
    n_samples: int = 3,
    max_body_chars: int = 240,
    min_body_chars: int = 80,
    before_date: dt.datetime | str | pd.Timestamp | None = None,
    skip_forwarded: bool = True,
) -> tuple[str, ...]:
    """Return up to ``n_samples`` formatted writing samples authored by ``name``.

    Parameters
    ----------
    name:
        Canonical sender name as it appears in ``sender_resolved``.
    parquet_path:
        Override path; default ``datasets/enron/processed/clean_emails.parquet``.
    n_samples:
        Target number of samples. Stratified by body-length quantile when
        the candidate pool is larger than ``n_samples``; otherwise all
        candidates are returned.
    max_body_chars:
        Each sample's body is truncated (with ellipsis) to this many
        characters so the resulting prompt stays within typical context
        windows even when ~6 personas are stitched.
    min_body_chars:
        Candidates with bodies shorter than this are skipped — too-short
        emails are usually one-liners or signatures and don't carry
        useful style signal.
    before_date:
        If set, only emails sent strictly before this timestamp are
        considered. Required for thread-continuation experiments to
        prevent hindsight leakage. Accepts ``datetime``, pandas
        ``Timestamp``, or any string parseable by
        ``pd.to_datetime``. ``None`` (default) admits all emails.

    Returns
    -------
    Tuple of formatted samples, each shaped like::

        Subject: {subject}
          {body, truncated}

    Empty tuple if the sender has no qualifying emails or the parquet
    file is missing.
    """
    p = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
    if not p.is_absolute():
        # Resolve relative paths against repo root (parents[3] from this file)
        p = Path(__file__).resolve().parents[3] / p
    if not p.exists():
        return ()

    cols = ["sender_resolved", "subject", "body_truncated", "body_chars"]
    if before_date is not None:
        cols.append("date")
    df = pd.read_parquet(p, columns=cols)
    sent = df[df["sender_resolved"] == name].copy()
    if sent.empty:
        return ()

    if before_date is not None:
        cutoff = pd.to_datetime(before_date, utc=True)
        sent_dates = pd.to_datetime(sent["date"], utc=True, errors="coerce")
        sent = sent[sent_dates < cutoff]
        if sent.empty:
            return ()

    sent = sent[sent["body_chars"] >= min_body_chars]
    if sent.empty:
        return ()

    if skip_forwarded:
        mask = sent["body_truncated"].fillna("").apply(
            lambda b: not _looks_forwarded(str(b))
        )
        sent = sent[mask]
        if sent.empty:
            return ()

    sent = sent.sort_values("body_chars").reset_index(drop=True)

    if len(sent) <= n_samples:
        idx = list(range(len(sent)))
    else:
        # Stratify by length quantile so we get a short / medium / long mix
        idx = [
            int(round((q / (n_samples + 1)) * (len(sent) - 1)))
            for q in range(1, n_samples + 1)
        ]

    out: list[str] = []
    for i in idx:
        row = sent.iloc[i]
        body = str(row.get("body_truncated") or "").strip()
        if not body:
            continue
        if len(body) > max_body_chars:
            body = body[: max_body_chars - 1].rstrip() + "…"
        # Indent body so the multi-line sample block is visually distinct
        body = body.replace("\n", "\n  ")
        subject = str(row.get("subject") or "").strip() or "(no subject)"
        out.append(f"Subject: {subject}\n  {body}")
    return tuple(out)
