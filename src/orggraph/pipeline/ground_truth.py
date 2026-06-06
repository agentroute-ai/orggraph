"""Generate ground truth CSV files from public Enron hierarchy data.

Sources:
- EnronData.org custodian names and titles
- Shetty & Adibi ex-employee status report
- Agarwal et al. (2012) methodology for dominance pair generation

Outputs (under ``--output-dir`` or ``orggraph.config.OUTPUT_DIR``):
- employees_ground_truth.csv
- dominance_pairs.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from orggraph.config import HIERARCHY_LEVELS, OUTPUT_DIR

ENRON_EMPLOYEES = [
    {"name": "Kenneth Lay", "email": "kenneth.lay@enron.com", "title": "Chairman and CEO", "level": "C-Suite"},
    {"name": "Jeffrey Skilling", "email": "jeff.skilling@enron.com", "title": "President and CEO", "level": "C-Suite"},
    {"name": "Andrew Fastow", "email": "andrew.fastow@enron.com", "title": "CFO", "level": "C-Suite"},
    {"name": "Richard Causey", "email": "richard.causey@enron.com", "title": "CAO", "level": "C-Suite"},
    {"name": "Steven Kean", "email": "steven.kean@enron.com", "title": "EVP and Chief of Staff", "level": "SVP"},
    {"name": "Mark Frevert", "email": "mark.frevert@enron.com", "title": "Vice Chairman", "level": "Vice Chairman"},
    {"name": "Greg Whalley", "email": "greg.whalley@enron.com", "title": "President and COO", "level": "C-Suite"},
    {"name": "Mark Haedicke", "email": "mark.haedicke@enron.com", "title": "Managing Director", "level": "VP"},
    {"name": "Jeff Dasovich", "email": "jeff.dasovich@enron.com", "title": "Government Relations Executive", "level": "Director"},
    {"name": "Sara Shackleton", "email": "sara.shackleton@enron.com", "title": "VP and Associate General Counsel", "level": "VP"},
    {"name": "Tana Jones", "email": "tana.jones@enron.com", "title": "Senior Legal Specialist", "level": "Manager"},
    {"name": "Vince Kaminski", "email": "vince.kaminski@enron.com", "title": "Managing Director, Research", "level": "VP"},
    {"name": "Louise Kitchen", "email": "louise.kitchen@enron.com", "title": "President, Enron Online", "level": "SVP"},
    {"name": "John Lavorato", "email": "john.lavorato@enron.com", "title": "CEO, Enron Americas", "level": "C-Suite"},
    {"name": "David Delainey", "email": "david.delainey@enron.com", "title": "CEO, Enron Energy Services", "level": "C-Suite"},
    {"name": "James Derrick", "email": "james.derrick@enron.com", "title": "General Counsel", "level": "SVP"},
    {"name": "Richard Sanders", "email": "richard.sanders@enron.com", "title": "VP and Assistant General Counsel", "level": "VP"},
    {"name": "Elizabeth Sager", "email": "elizabeth.sager@enron.com", "title": "VP and Assistant General Counsel", "level": "VP"},
    {"name": "Gerald Nemec", "email": "gerald.nemec@enron.com", "title": "Director", "level": "Director"},
    {"name": "Chris Germany", "email": "chris.germany@enron.com", "title": "Pipeline Trading Manager", "level": "Manager"},
    {"name": "Mike Grigsby", "email": "mike.grigsby@enron.com", "title": "VP, Gas Trading", "level": "VP"},
    {"name": "Scott Neal", "email": "scott.neal@enron.com", "title": "VP, Trading", "level": "VP"},
    {"name": "Barry Tycholiz", "email": "barry.tycholiz@enron.com", "title": "VP, Gas Trading", "level": "VP"},
    {"name": "Susan Scott", "email": "susan.scott@enron.com", "title": "VP", "level": "VP"},
    {"name": "Kay Mann", "email": "kay.mann@enron.com", "title": "Senior Counsel", "level": "Manager"},
    {"name": "Matthew Lenhart", "email": "matthew.lenhart@enron.com", "title": "Trader", "level": "Employee"},
    {"name": "Phillip Love", "email": "phillip.love@enron.com", "title": "Analyst", "level": "Employee"},
    {"name": "Tracy Geaccone", "email": "tracy.geaccone@enron.com", "title": "VP, Pipeline Operations", "level": "VP"},
    {"name": "Danny McCarty", "email": "danny.mccarty@enron.com", "title": "VP", "level": "VP"},
    {"name": "Bill Williams", "email": "bill.williams@enron.com", "title": "Trader", "level": "Employee"},
]


def generate_ground_truth(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(ENRON_EMPLOYEES)
    df["level_numeric"] = df["level"].map(HIERARCHY_LEVELS)

    df.to_csv(output_dir / "employees_ground_truth.csv", index=False)
    print(f"Saved {len(df)} employees to {output_dir / 'employees_ground_truth.csv'}")

    # Generate dominance pairs: person at level N dominates all persons at level < N
    pairs = []
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and row1["level_numeric"] > row2["level_numeric"]:
                pairs.append({
                    "superior": row1["name"],
                    "subordinate": row2["name"],
                    "superior_level": row1["level"],
                    "subordinate_level": row2["level"],
                })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(output_dir / "dominance_pairs.csv", index=False)
    print(f"Generated {len(df_pairs)} dominance pairs to {output_dir / 'dominance_pairs.csv'}")

    return df, df_pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for output CSVs (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)
    generate_ground_truth(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
