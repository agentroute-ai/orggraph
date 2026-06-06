"""Stage 7.5 entry point — inspect the V1 KG-as-tool catalog.

Usage:
    python -m orggraph.pipeline.agents               # list catalog
    python -m orggraph.pipeline.agents --json        # full JSON schema
"""

from __future__ import annotations

import argparse
import json

from orggraph.pipeline.agents.tools import build_default_registry


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--json", action="store_true",
                   help="Print full OpenAI tools schema as JSON.")
    args = p.parse_args(argv)

    # The factory takes a driver but doesn't use it at registration time.
    reg = build_default_registry(driver=None)

    if args.json:
        print(json.dumps(reg.to_openai_tools(), indent=2))
        return

    print(f"V1 KG-as-tool catalog ({len(reg)} tools):\n")
    for t in reg:
        print(f"  {t.name}")
        print(f"    {t.description[:80]}")
        required = t.args_schema.get("required", [])
        params = list(t.args_schema.get("properties", {}))
        print(f"    args: required={required}, all={params}\n")


if __name__ == "__main__":
    main()
