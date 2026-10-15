#!/usr/bin/env python3
"""
Model Benchmark Suite for Raspberry Pi 5 + Ollama
Benchmarks local GGUF models: latency, throughput, memory usage.
Stores results in SQLite for historical comparison.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_API = os.environ.get("OLLAMA_API", "http://localhost:11434")
DB_PATH = os.environ.get("BENCHMARK_DB", str(Path.home() / ".local/share/model-benchmark/results.db"))
API_RETRIES = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str
    prompt: str
    created_at: str
    total_time_s: float
    time_to_first_token_s: float
    tokens_generated: int
    tokens_per_sec: float
    prompt_eval_time_s: float
    eval_count: int
    peak_memory_mb: float
    temperature: float
    num_ctx: int


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def init_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            total_time_s REAL NOT NULL,
            time_to_first_token_s REAL NOT NULL,
            tokens_generated INTEGER NOT NULL,
            tokens_per_sec REAL NOT NULL,
            prompt_eval_time_s REAL,
            eval_count INTEGER,
            peak_memory_mb REAL,
            temperature REAL DEFAULT 0.7,
            num_ctx INTEGER DEFAULT 4096
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            name TEXT PRIMARY KEY,
            size_bytes INTEGER,
            downloaded_at TEXT,
            source TEXT,
            parameters TEXT,
            quant_method TEXT
        )
    """)
    conn.commit()
    return conn


def save_result(conn: sqlite3.Connection, result: BenchmarkResult):
    conn.execute(
        """INSERT INTO benchmarks
           (model, prompt, created_at, total_time_s, time_to_first_token_s,
            tokens_generated, tokens_per_sec, prompt_eval_time_s, eval_count,
            peak_memory_mb, temperature, num_ctx)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (result.model, result.prompt, result.created_at, result.total_time_s,
         result.time_to_first_token_s, result.tokens_generated,
         result.tokens_per_sec, result.prompt_eval_time_s, result.eval_count,
         result.peak_memory_mb, result.temperature, result.num_ctx),
    )
    conn.commit()


def save_model_info(conn: sqlite3.Connection, name: str, info: dict):
    conn.execute(
        """INSERT OR REPLACE INTO models
           (name, size_bytes, downloaded_at, source, parameters, quant_method)
           VALUES (?,?,?,?,?,?)""",
        (name, info.get("size", 0), datetime.now(timezone.utc).isoformat(),
         info.get("source", "ollama"), info.get("parameters", ""),
         info.get("quant_method", "")),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------

def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with retry logic."""
    last_error: Exception | None = None
    for attempt in range(API_RETRIES + 1):
        try:
            r = requests.request(method, url, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last_error = e
            if attempt < API_RETRIES:
                time.sleep(1)
    raise last_error  # type: ignore[misc]


def list_ollama_models() -> list[dict]:
    r = _request_with_retry("GET", f"{OLLAMA_API}/api/tags", timeout=10)
    return r.json().get("models", [])


def get_model_info(name: str) -> dict:
    r = _request_with_retry("POST", f"{OLLAMA_API}/api/show", json={"name": name}, timeout=10)
    return r.json()


def get_ollama_memory_mb() -> float:
    """Get current ollama process memory in MB (VmRSS sum)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama"], capture_output=True, text=True, timeout=5
        )
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if not pids:
            return 0.0
        total = 0.0
        for pid in pids:
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1]) / 1024  # kB -> MB
                            break
            except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                continue
        return total
    except Exception:
        return 0.0


def download_model(name: str) -> bool:
    """Pull a model into Ollama."""
    print(f"Downloading model: {name}...")
    try:
        r = _request_with_retry(
            "POST",
            f"{OLLAMA_API}/api/pull",
            json={"name": name},
            stream=True,
            timeout=600,
        )
        for line in r.iter_lines():
            if line:
                try:
                    progress = json.loads(line)
                    if "completed" in progress and "total" in progress:
                        pct = progress["completed"] / progress["total"] * 100
                        print(f"  {pct:.1f}%", end="\r")
                except (json.JSONDecodeError, ZeroDivisionError):
                    pass
        print(f"\nDownloaded: {name}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

STANDARD_PROMPTS = [
    {
        "name": "short-qa",
        "prompt": "What is the capital of France?",
        "temperature": 0.7,
        "num_ctx": 2048,
    },
    {
        "name": "medium-reasoning",
        "prompt": "Explain the difference between TCP and UDP networking protocols, and give an example use case for each.",
        "temperature": 0.7,
        "num_ctx": 4096,
    },
    {
        "name": "long-generation",
        "prompt": "Write a short story about a robot learning to paint. Keep it under 200 words.",
        "temperature": 0.9,
        "num_ctx": 4096,
    },
    {
        "name": "coding",
        "prompt": "Write a Python function that takes a list of integers and returns the sum of all even numbers in the list.",
        "temperature": 0.2,
        "num_ctx": 4096,
    },
]


def run_benchmark(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_ctx: int = 4096,
    max_tokens: int = 512,
) -> BenchmarkResult:
    """Run a single benchmark: measure TTFT, total time, tokens/sec, memory."""
    # Input validation
    if temperature < 0:
        temperature = 0.0
    if temperature > 2.0:
        temperature = 2.0
    if num_ctx < 1:
        num_ctx = 1
    if num_ctx > 131072:
        num_ctx = 131072
    if max_tokens < 1:
        max_tokens = 1
    if max_tokens > 8192:
        max_tokens = 8192

    print(f"  Running: {model} | temp={temperature} ctx={num_ctx}")

    mem_before = get_ollama_memory_mb()
    token_count = 0
    first_token_time: float | None = None
    start_time = time.perf_counter()
    total_time = 0.0
    eval_count = 0
    prompt_eval_time = 0.0
    timed_out = False

    try:
        r = _request_with_retry(
            "POST",
            f"{OLLAMA_API}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": max_tokens,
                },
            },
            stream=True,
            timeout=120,
        )

        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()

            if first_token_time is None and chunk.get("response"):
                first_token_time = now
            if chunk.get("response"):
                token_count += 1
            if chunk.get("done"):
                total_time = now - start_time
                eval_count = chunk.get("eval_count", 0)
                prompt_eval_time = chunk.get("prompt_eval_duration", 0) / 1e9
                break

    except requests.Timeout:
        print(f"    TIMEOUT after 120s")
        total_time = time.perf_counter() - start_time
        timed_out = True
    except Exception as e:
        print(f"    ERROR: {e}")
        total_time = time.perf_counter() - start_time

    if not timed_out and total_time < 0.001:
        total_time = time.perf_counter() - start_time

    mem_after = get_ollama_memory_mb()
    # Use delta for more accurate per-benchmark memory measurement
    memory_delta = max(mem_after - mem_before, 0.0)
    # If delta is 0 (process already running), report current usage
    peak_memory = memory_delta if memory_delta > 0 else mem_after

    ttft = (first_token_time - start_time) if first_token_time else -1.0
    total_time = max(total_time, 0.001)  # avoid div-by-zero
    tps = token_count / total_time if total_time > 0 else 0.0

    ttft_str = f"{ttft:.2f}s" if ttft >= 0 else "N/A"
    print(f"    TTFT: {ttft_str} | Total: {total_time:.2f}s | "
          f"Tokens: {token_count} | {tps:.1f} tok/s | Mem: {peak_memory:.0f} MB")

    return BenchmarkResult(
        model=model,
        prompt=prompt,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_time_s=round(total_time, 4),
        time_to_first_token_s=round(ttft, 4) if ttft >= 0 else -1.0,
        tokens_generated=token_count,
        tokens_per_sec=round(tps, 2),
        prompt_eval_time_s=round(prompt_eval_time, 4),
        eval_count=eval_count,
        peak_memory_mb=round(peak_memory, 1),
        temperature=temperature,
        num_ctx=num_ctx,
    )


def run_all_benchmarks(model: str, prompts: list[dict] | None = None) -> list[BenchmarkResult]:
    prompts = prompts or STANDARD_PROMPTS
    results = []
    for p in prompts:
        result = run_benchmark(
            model=model,
            prompt=p["prompt"],
            temperature=p.get("temperature", 0.7),
            num_ctx=p.get("num_ctx", 4096),
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(conn: sqlite3.Connection, model: str | None = None):
    """Print a formatted report of benchmark results."""
    query = "SELECT * FROM benchmarks"
    params: tuple = ()
    if model:
        query += " WHERE model = ?"
        params = (model,)
    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("No benchmark results found.")
        return

    print("\n" + "=" * 100)
    print(f"{'Model':<30} {'Date':<12} {'TTFT':>8} {'Total':>8} {'Tok/s':>8} {'Memory':>10}")
    print("-" * 100)
    for row in rows:
        # id, model, prompt, created_at, total_time, ttft, tokens, tps, prompt_eval, eval_count, mem, temp, ctx
        ttft = f"{row[5]:.2f}s" if row[5] >= 0 else "N/A"
        print(f"{row[1]:<30} {row[3][:10]:<12} {ttft:>8} {row[4]:>7.2f}s "
              f"{row[7]:>7.1f} {row[10]:>8.0f}MB")
    print("=" * 100)


def print_model_comparison(conn: sqlite3.Connection):
    """Compare all models side by side."""
    rows = conn.execute("""
        SELECT model,
               COUNT(*) as runs,
               AVG(tokens_per_sec) as avg_tps,
               MIN(tokens_per_sec) as min_tps,
               MAX(tokens_per_sec) as max_tps,
               AVG(time_to_first_token_s) as avg_ttft,
               AVG(peak_memory_mb) as avg_mem
        FROM benchmarks
        GROUP BY model
        ORDER BY avg_tps DESC
    """).fetchall()

    if not rows:
        print("No benchmark results found.")
        return

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("-" * 90)
    print(f"{'Model':<30} {'Runs':>5} {'Avg tok/s':>10} {'Min':>8} {'Max':>8} {'Avg TTFT':>10} {'Avg Mem':>10}")
    print("-" * 90)
    for row in rows:
        print(f"{row[0]:<30} {row[1]:>5} {row[2]:>9.1f} {row[3]:>7.1f} {row[4]:>7.1f} "
              f"{row[5]:>9.2f}s {row[6]:>8.0f}MB")
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model Benchmark Suite for Ollama")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List models in Ollama")

    dl = sub.add_parser("download", help="Download a model")
    dl.add_argument("name", help="Model name (e.g. llama3.1:8b-instruct)")

    bench = sub.add_parser("bench", help="Run benchmarks")
    bench.add_argument("model", help="Model name")
    bench.add_argument("--prompt", action="append", help="Custom prompt (can repeat)")
    bench.add_argument("--temperature", type=float, default=0.7)
    bench.add_argument("--num-ctx", type=int, default=4096)
    bench.add_argument("--max-tokens", type=int, default=512)

    rep = sub.add_parser("report", help="Show benchmark report")
    rep.add_argument("--model", help="Filter by model")

    sub.add_parser("compare", help="Compare all models")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    conn = init_db(DB_PATH)

    if args.command == "list":
        models = list_ollama_models()
        if not models:
            print("No models found in Ollama.")
        else:
            print(f"\n{'Name':<40} {'Size':>12} {'Format':>10}")
            print("-" * 65)
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                print(f"{m['name']:<40} {size_gb:>10.2f}GB {m.get('format',''):>10}")

    elif args.command == "download":
        if download_model(args.name):
            info = get_model_info(args.name)
            save_model_info(conn, args.name, info)
            print(f"Saved model info for {args.name}")

    elif args.command == "bench":
        if args.prompt:
            prompts = [{"name": f"custom-{i}", "prompt": p, "temperature": args.temperature, "num_ctx": args.num_ctx}
                       for i, p in enumerate(args.prompt)]
        else:
            prompts = None

        results = run_all_benchmarks(args.model, prompts)
        for r in results:
            save_result(conn, r)
        print(f"\nSaved {len(results)} benchmark results to {DB_PATH}")

    elif args.command == "report":
        print_report(conn, args.model)

    elif args.command == "compare":
        print_model_comparison(conn)

    conn.close()


if __name__ == "__main__":
    main()
