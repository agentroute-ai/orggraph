"""Load Enron email data from HuggingFace and local employee records."""

import json

import pandas as pd

from orggraph.config import EMPLOYEES_JSON, HF_DATASET


def load_employees() -> pd.DataFrame:
    """Load the 148 Enron employees from employees.json."""
    with open(EMPLOYEES_JSON) as f:
        data = json.load(f)

    records = []
    for emp in data:
        records.append({
            "name": emp["name"],
            "given_name": emp.get("givenName", ""),
            "family_name": emp.get("familyName", ""),
            "email": emp["email"],
            "folder_name": emp.get("alternateName", ""),
            "additional_name": emp.get("additionalName", ""),
        })

    return pd.DataFrame(records)


def load_emails(limit: int | None = None) -> pd.DataFrame:
    """Load pre-parsed Enron emails from HuggingFace.

    Args:
        limit: If set, only load this many emails (for development).

    Returns:
        DataFrame with columns from the HuggingFace dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    df = ds.to_pandas()
    return df
