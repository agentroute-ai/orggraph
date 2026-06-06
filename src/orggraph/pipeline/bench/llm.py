"""Benchmark an OpenAI-compatible LLM endpoint at increasing concurrency.

For each concurrency level N in --concurrencies (default 8,16,32,64,128,256),
issues `--n-prompts` realistic Stage 2a calls in parallel, measures:

  - aggregate calls/sec (throughput)
  - aggregate tokens/sec (prompt+generation)
  - per-call wall latency: mean, p50, p95
  - vLLM saturation indicators (num_requests_waiting peak,
    kv_cache_usage_perc peak, num_preemptions_total delta) when the
    server exposes /metrics
  - error rate

Output: pretty table to stdout, CSV at `--output`. Use the table to pick
the highest concurrency level whose latency p95 hasn't blown up and whose
queue stayed at zero — that's your saturation point.

The benchmark uses `extract_email_signals.build_prompt` so the workload
matches Stage 2a's actual prompts byte-for-byte (including the cluster
context block when Stage 2b artefacts are present).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from orggraph.config import OUTPUT_DIR
from orggraph.llm.client import LLMClient
from orggraph.pipeline.signals import extract as _EXTR


# ---------------------------------------------------------------------------
# Helpers — deterministic / unit-testable
# ---------------------------------------------------------------------------


def percentile(xs: list[float], p: float) -> float:
    """p in [0, 100]. Returns NaN for empty input."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def summarise_latencies(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "min", "max")}
    return {
        "mean": statistics.mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
    }


def fetch_vllm_metrics(base_url: str) -> dict[str, float]:
    """Read prometheus metrics from /metrics if available; return empty dict otherwise."""
    if not base_url.endswith("/v1"):
        return {}
    metrics_url = base_url[:-3] + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=2) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}
    out: dict[str, float] = {}
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    }
    for line in text.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        name = line.split("{", 1)[0]
        if name not in wanted:
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        # Last-write-wins; ok for our gauges/counters
        out[name] = value
    return out


def load_sample_prompts(n: int) -> list[str]:
    """Build N realistic Stage 2a prompts from clean_emails.parquet."""
    parquet = OUTPUT_DIR / "clean_emails.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing {parquet}; run filter_corpus.py first.")
    df = pd.read_parquet(parquet).head(max(n * 2, 50))

    # Pull cluster context if Stage 2b artefacts exist (zero cost otherwise)
    try:
        ctx_map = _EXTR.load_cluster_context_map(df)
    except Exception:
        ctx_map = {}

    prompts: list[str] = []
    for _, row in df.iterrows():
        if len(prompts) >= n:
            break
        subj = (row.get("subject") or "").strip()
        body = (row.get("body_truncated") or "").strip()[:8000]
        ctx = ctx_map.get(row["email_id"])
        prompts.append(_EXTR.build_prompt(subj, body, cluster_context=ctx))
    return prompts


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def bench_one_level(
    clients: list[LLMClient],
    model: str,
    prompts: list[str],
    concurrency: int,
    base_urls: list[str],
) -> dict[str, float]:
    """Issue len(prompts) calls at the given concurrency; return summary.

    Calls are round-robin'd across `clients`. vLLM metrics are aggregated
    across `base_urls` (peaks and counter deltas summed across endpoints).
    """
    pre = {u: fetch_vllm_metrics(u) for u in base_urls}
    latencies: list[float] = []
    n_errors = 0
    peak_waiting = sum(p.get("vllm:num_requests_waiting", 0.0) for p in pre.values())
    peak_running = sum(p.get("vllm:num_requests_running", 0.0) for p in pre.values())
    peak_kv = max(
        (p.get("vllm:kv_cache_usage_perc", 0.0) for p in pre.values()),
        default=0.0,
    )

    rr_counter = itertools.count()

    def pick_client() -> LLMClient:
        return clients[next(rr_counter) % len(clients)]

    def call_one(p: str) -> tuple[float, bool]:
        client = pick_client()
        t0 = time.perf_counter()
        try:
            client.json_chat(model=model, prompt=p)
            return time.perf_counter() - t0, True
        except Exception:
            return time.perf_counter() - t0, False

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_one, p) for p in prompts]
        # Sample metrics every ~0.5s for peaks (aggregate across endpoints)
        last_sample = 0.0
        for fut in as_completed(futs):
            now = time.perf_counter()
            if now - last_sample > 0.5:
                snap_running = 0.0
                snap_waiting = 0.0
                snap_kv = 0.0
                for u in base_urls:
                    m = fetch_vllm_metrics(u)
                    snap_running += m.get("vllm:num_requests_running", 0.0)
                    snap_waiting += m.get("vllm:num_requests_waiting", 0.0)
                    snap_kv = max(snap_kv, m.get("vllm:kv_cache_usage_perc", 0.0))
                peak_running = max(peak_running, snap_running)
                peak_waiting = max(peak_waiting, snap_waiting)
                peak_kv = max(peak_kv, snap_kv)
                last_sample = now
            lat, ok = fut.result()
            latencies.append(lat)
            if not ok:
                n_errors += 1
    wall = time.perf_counter() - t_start

    post = {u: fetch_vllm_metrics(u) for u in base_urls}
    prompt_tok = sum(
        post[u].get("vllm:prompt_tokens_total", 0.0) - pre[u].get("vllm:prompt_tokens_total", 0.0)
        for u in base_urls
    )
    gen_tok = sum(
        post[u].get("vllm:generation_tokens_total", 0.0) - pre[u].get("vllm:generation_tokens_total", 0.0)
        for u in base_urls
    )
    preempts = sum(
        post[u].get("vllm:num_preemptions_total", 0.0) - pre[u].get("vllm:num_preemptions_total", 0.0)
        for u in base_urls
    )

    n_total = len(prompts)
    n_ok = n_total - n_errors
    stats = summarise_latencies([l for l, in zip(latencies)])  # noqa: E741

    return {
        "concurrency": concurrency,
        "n_calls": n_total,
        "n_errors": n_errors,
        "wall_s": wall,
        "calls_per_sec": n_ok / wall if wall > 0 else 0.0,
        "prompt_tokens": prompt_tok,
        "generation_tokens": gen_tok,
        "tokens_per_sec": (prompt_tok + gen_tok) / wall if wall > 0 else 0.0,
        "lat_mean_s": stats["mean"],
        "lat_p50_s": stats["p50"],
        "lat_p95_s": stats["p95"],
        "lat_p99_s": stats["p99"],
        "peak_running": peak_running,
        "peak_waiting": peak_waiting,
        "peak_kv_pct": peak_kv * 100,
        "preemptions_delta": preempts,
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        ("concurrency", 5),
        ("n_calls", 7),
        ("wall_s", 8),
        ("calls_per_sec", 9),
        ("tokens_per_sec", 11),
        ("lat_mean_s", 9),
        ("lat_p95_s", 8),
        ("peak_running", 8),
        ("peak_waiting", 8),
        ("peak_kv_pct", 9),
        ("n_errors", 5),
    ]
    header = " ".join(f"{c:>{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " ".join(
            (
                f"{int(r[c]):>{w}d}"
                if c in {"concurrency", "n_calls", "n_errors"}
                else f"{r[c]:>{w}.2f}"
            )
            for c, w in cols
        )
        print(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Concurrency-sweep benchmark for the Stage 2a LLM endpoint."
    )
    parser.add_argument(
        "--base-urls",
        default=os.environ.get(
            "OPENAI_BASE_URLS",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        ),
        help="Comma-separated list of OpenAI-compatible endpoints; round-robined.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get(
        "INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"))
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="8,16,32,64,128,256",
        help="Comma-separated concurrency levels to test.",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=50,
        help="Calls per concurrency level. Larger = more stable.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(OUTPUT_DIR / "bench_llm_results.csv"),
    )
    parser.add_argument(
        "--stop-on-saturation", action="store_true",
        help="Stop when peak_waiting > 0 (queue forming).",
    )
    args = parser.parse_args(argv)

    levels = [int(x) for x in args.concurrencies.split(",") if x]
    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    print(f"[backend] {len(base_urls)} endpoint(s):")
    for u in base_urls:
        print(f"          - {u}")
    print(f"[model]   {args.model}")
    print(f"[plan]    concurrencies={levels}, prompts/level={args.n_prompts}")

    print("[1/3] Building prompts (matches Stage 2a payload)...")
    prompts = load_sample_prompts(args.n_prompts)
    print(f"      {len(prompts)} prompts ready (avg {sum(len(p) for p in prompts)//len(prompts)} chars)")

    print("[2/3] Sweeping concurrency levels...")
    clients = [LLMClient(base_url=u, api_key=args.api_key) for u in base_urls]
    results: list[dict] = []
    for c in levels:
        print(f"\n  [c={c}] dispatching {len(prompts)} calls...")
        row = bench_one_level(clients, args.model, prompts, c, base_urls)
        results.append(row)
        print(
            f"  [c={c}] wall={row['wall_s']:.1f}s  "
            f"calls/s={row['calls_per_sec']:.2f}  "
            f"tok/s={row['tokens_per_sec']:.0f}  "
            f"p95={row['lat_p95_s']:.2f}s  "
            f"running={int(row['peak_running'])}  "
            f"waiting={int(row['peak_waiting'])}  "
            f"kv={row['peak_kv_pct']:.1f}%  "
            f"errors={int(row['n_errors'])}"
        )
        if args.stop_on_saturation and row["peak_waiting"] > 0:
            print(f"  [stop] saturation detected at c={c} (peak_waiting={int(row['peak_waiting'])})")
            break

    print("\n[3/3] Summary:")
    print_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_path}")

    # Recommend a working point
    saturated = [r for r in results if r["peak_waiting"] > 0]
    healthy = [r for r in results if r["peak_waiting"] == 0 and r["n_errors"] == 0]
    if healthy:
        best = max(healthy, key=lambda r: r["calls_per_sec"])
        print(
            f"\nRecommendation: --workers {int(best['concurrency'])} "
            f"({best['calls_per_sec']:.2f} calls/s, p95 {best['lat_p95_s']:.2f}s, "
            f"no queueing, no errors)."
        )
    if saturated:
        print(
            f"Saturation point: c={int(saturated[0]['concurrency'])} "
            f"(queue forming). Going higher will queue, not parallelise further."
        )


if __name__ == "__main__":
    main()
