"""Paths and constants for the OrgGraph pipeline."""

from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATASETS_DIR = REPO_ROOT / "datasets" / "enron"
EMPLOYEES_JSON = DATASETS_DIR / "employees.json"
OUTPUT_DIR = DATASETS_DIR / "processed"

# HuggingFace dataset
HF_DATASET = "corbt/enron-emails"

# Hierarchy tiers (numeric for ordering)
HIERARCHY_LEVELS = {
    "C-Suite": 6,
    "Vice Chairman": 5,
    "SVP": 4,
    "VP": 3,
    "Director": 2,
    "Manager": 1,
    "Employee": 0,
    "Assistant": 0,
}
