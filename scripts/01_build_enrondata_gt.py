"""Build a 103-person ground truth from EnronData.org's public custodian-titles file.

Source: https://github.com/enrondata/enrondata/raw/master/data/misc/edo_enron-custodians-data.html
License: Creative Commons Attribution 3.0 United States

Maps the 10 distinct titles in the file (CEO, President, Managing Director,
Vice President, Director, Manager, In House Lawyer, Trader, Employee, N/A) to
the seven hierarchy tiers in ``orggraph.config.HIERARCHY_LEVELS``. Custodians
with ``N/A`` titles are excluded from the GT but remain in the extracted graph.

Outputs (under ``datasets/enron/processed/``):
  - employees_ground_truth_v2.csv  : the 103 mappable custodians with tier + level
  - dominance_pairs_v2.csv          : all cross-tier pairs (current run: 3,814)

The 30-person hand-curated GT in ``orggraph.pipeline.ground_truth`` is preserved
as a sensitivity-check eval; this v2 GT is used as the headline for RQ1.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orggraph.config import HIERARCHY_LEVELS  # noqa: E402

EDO_URL = (
    "https://github.com/enrondata/enrondata/raw/master/data/misc/"
    "edo_enron-custodians-data.html"
)
EDO_LOCAL = ROOT / "datasets/enron/enrondata_custodians.html"

PROCESSED = ROOT / "datasets/enron/processed"
EMPLOYEES_OUT = PROCESSED / "employees_ground_truth_v2.csv"
PAIRS_OUT = PROCESSED / "dominance_pairs_v2.csv"

TITLE_TO_TIER: dict[str, str] = {
    "CEO": "C-Suite",
    "President": "C-Suite",
    "Managing Director": "SVP",
    "Vice President": "VP",
    "Director": "Director",
    "Manager": "Manager",
    "In House Lawyer": "Manager",
    "Trader": "Employee",
    "Employee": "Employee",
}

TR_RE = re.compile(
    r"<tr>\s*<td>(\d+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]*)</td>",
    re.S,
)


def fetch_html() -> str:
    """Return the HTML, downloading if not cached locally."""
    if not EDO_LOCAL.exists():
        EDO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {EDO_URL}")
        with urllib.request.urlopen(EDO_URL) as r:
            EDO_LOCAL.write_bytes(r.read())
    return EDO_LOCAL.read_text()


def parse_custodians(html: str) -> pd.DataFrame:
    rows = [
        {
            "id": int(m.group(1)),
            "folder": m.group(2).strip(),
            "name_enrondata": m.group(3).strip(),
            "title_raw": m.group(4).strip(),
        }
        for m in TR_RE.finditer(html)
    ]
    return pd.DataFrame(rows)


def map_to_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["title_raw"].map(TITLE_TO_TIER)
    df["level_numeric"] = df["tier"].map(HIERARCHY_LEVELS)
    return df.dropna(subset=["level_numeric"]).reset_index(drop=True)


def attach_canonical_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``name_enrondata`` with the canonical ``name`` from employees.json
    via the maildir folder, so rows align with our extracted graph."""
    employees_json = ROOT / "datasets/enron/employees.json"
    with open(employees_json) as f:
        ej = json.load(f)
    folder_to_canonical = {
        e.get("alternateName", ""): e["name"] for e in ej if e.get("alternateName")
    }
    df = df.copy()
    df["name"] = df["folder"].map(folder_to_canonical)
    matched = df.dropna(subset=["name"]).reset_index(drop=True)
    if len(matched) < len(df):
        print(
            f"  WARNING: {len(df) - len(matched)} EnronData rows have no folder "
            f"match in employees.json"
        )
    return matched


def generate_pairs(employees: pd.DataFrame) -> pd.DataFrame:
    """All directed (superior, subordinate) cross-tier pairs."""
    pairs = []
    rows = list(employees.itertuples(index=False))
    for sup in rows:
        for sub in rows:
            if sup.name == sub.name:
                continue
            if sup.level_numeric > sub.level_numeric:
                pairs.append(
                    {
                        "superior": sup.name,
                        "subordinate": sub.name,
                        "superior_level": sup.tier,
                        "subordinate_level": sub.tier,
                    }
                )
    return pd.DataFrame(pairs)


def main() -> None:
    html = fetch_html()
    custodians = parse_custodians(html)
    print(f"Parsed {len(custodians)} custodians from {EDO_LOCAL.name}")

    mappable = map_to_tiers(custodians)
    print(f"With mappable hierarchy title: {len(mappable)} / {len(custodians)}")

    employees = attach_canonical_names(mappable)
    print(f"After canonical-name match via employees.json: {len(employees)}")
    print("\nTier distribution:")
    print(employees["tier"].value_counts().to_string())

    pairs = generate_pairs(employees)
    print(f"\nGenerated {len(pairs)} cross-tier dominance pairs")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cols_emp = ["name", "folder", "tier", "level_numeric", "title_raw"]
    employees[cols_emp].to_csv(EMPLOYEES_OUT, index=False)
    pairs.to_csv(PAIRS_OUT, index=False)
    print(f"\nSaved: {EMPLOYEES_OUT}")
    print(f"Saved: {PAIRS_OUT}")


if __name__ == "__main__":
    main()
