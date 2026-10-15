#!/usr/bin/env python3
"""
Accuracy evaluation for coding models.
Tests each model against known problems and grades responses.

Security: All model-generated code runs in a sandboxed subprocess with
AST validation, CPU/memory limits, and timeout. Never exec() directly.
"""

import ast
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_API = os.environ.get("OLLAMA_API", "http://localhost:11434")
DB_PATH = os.environ.get("BENCHMARK_DB", str(Path.home() / ".local/share/model-benchmark/accuracy.db"))
EXEC_TIMEOUT = 5  # seconds for sandboxed code execution
API_RETRIES = 2

# Imports that are dangerous or unnecessary for code grading
DANGEROUS_IMPORTS = {
    "os", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "importlib.util", "importlib.machinery",
    "ctypes", "signal", "threading", "multiprocessing", "asyncio",
    "urllib", "http", "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "pickle", "marshal", "shelve", "dbm", "sqlite3", "requests",
    "webbrowser", "pty", "tty", "fcntl", "mmap", "resource",
    "gc", "inspect", "code", "codeop", "compileall", "py_compile",
    "sysconfig", "distutils", "ensurepip", "venv", "zipfile",
    "tarfile", "gzip", "bz2", "lzma", "zlib", "crypt", "pwd",
    "grp", "spwd", "termios", "posix", "nt", "sys",
    # Attribute/introspection and low-level escape vectors
    "getattr", "setattr", "delattr", "hasattr", "builtins", "io",
    "struct", "codecs", "operator", "functools", "itertools",
    "collections", "array", "select",
}

# Builtins that must never be called by graded code: they either execute
# arbitrary code (eval/exec/compile/__import__) or expose the interpreter
# internals needed to climb out of the sandbox (getattr/globals/...).
DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "breakpoint",
    "locals", "vars", "globals", "getattr", "setattr", "delattr",
    "hasattr", "exit", "quit", "input",
}


@dataclass
class AccuracyResult:
    model: str
    test_name: str
    category: str
    prompt: str
    response: str
    correct: bool
    score: float  # 0.0 - 1.0
    reasoning: str
    created_at: str
    response_time_s: float


# ---------------------------------------------------------------------------
# Sandboxed code execution
# ---------------------------------------------------------------------------

def _is_code_safe(code: str) -> bool:
    """AST check: reject dangerous imports and sandbox escape vectors."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DANGEROUS_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in DANGEROUS_IMPORTS:
                return False
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                return False
    return True


def run_sandboxed(code: str, timeout: int = EXEC_TIMEOUT) -> tuple[bool, str, str]:
    """Run code in a restricted subprocess. Returns (success, stdout, stderr)."""
    if not _is_code_safe(code):
        return False, "", "Code contains dangerous imports"

    wrapper = f"""import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
exec({code!r})
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
            env={"PATH": ""},
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", f"Sandbox error: {e}"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_python_code(response: str) -> str | None:
    """Extract Python code from a response that may contain markdown."""
    # Try ```python ... ``` first
    if "```python" in response:
        parts = response.split("```python")
        for part in parts[1:]:
            end = part.find("```")
            if end >= 0:
                code = part[:end].strip()
                if code:
                    return code

    # Try generic ``` ... ```
    if "```" in response:
        parts = response.split("```")
        for i in range(1, len(parts), 2):  # odd indices are code blocks
            code = parts[i].strip()
            if code:
                # Remove language identifier on first line
                lines = code.split("\n")
                if lines[0].strip() in ("python", "python3", "py"):
                    code = "\n".join(lines[1:]).strip()
                if code:
                    return code

    # Fallback: find def/import/class at start of line
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if not in_code:
            if stripped.startswith(("def ", "import ", "from ", "class ", "if __name__")):
                in_code = True
                code_lines.append(line)
        else:
            if stripped == "" and code_lines:
                break
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    return None


# ---------------------------------------------------------------------------
# Grading functions
# ---------------------------------------------------------------------------

def grade_fibonacci(response: str) -> tuple[bool, float, str]:
    """Check if response contains a correct Fibonacci implementation."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            # Probe the function by name. Note: globals()/eval/vars are blocked
            # by the sandbox (and this probe is AST-checked too), so reference
            # the candidate names directly rather than introspecting the module.
            probe = code + """

_fib = fib if 'fib' in dir() else fibonacci if 'fibonacci' in dir() else None
print(_fib(10) if _fib else None)
"""
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2:
                try:
                    val = int(stdout2.strip())
                    if val == 55:
                        return True, 1.0, "Code execution: fib(10)=55"
                except ValueError:
                    pass

    # Heuristic fallback
    response_lower = response.lower()
    has_def = "def" in response_lower and "fib" in response_lower
    has_loop_or_recursion = any(kw in response_lower for kw in ["for", "while", "return"])
    has_base_case = any(kw in response_lower for kw in ["if", "base", "n <= 1", "n < 2", "n == 0", "n == 1"])
    score = 0.0
    if has_def: score += 0.3
    if has_loop_or_recursion: score += 0.3
    if has_base_case: score += 0.4
    return score >= 0.7, score, f"Heuristic: def={has_def}, loop/recursion={has_loop_or_recursion}, base_case={has_base_case}"


def grade_reverse_string(response: str) -> tuple[bool, float, str]:
    """Check if response contains a correct string reversal."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            probe = code + "\n\n_r = reverse_string('hello') if 'reverse_string' in dir() else reverse('hello') if 'reverse' in dir() else rev('hello') if 'rev' in dir() else None\nprint(_r)"
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2 and stdout2.strip() == "olleh":
                return True, 1.0, "Code execution: reverse_string('hello')='olleh'"

    response_lower = response.lower()
    has_def = "def" in response_lower
    has_reverse = any(kw in response_lower for kw in ["[::-1]", "reversed", "reverse"])
    score = 0.0
    if has_def: score += 0.4
    if has_reverse: score += 0.6
    return score >= 0.7, score, f"Heuristic: def={has_def}, reverse_logic={has_reverse}"


def grade_binary_search(response: str) -> tuple[bool, float, str]:
    """Check if response contains a correct binary search implementation."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            probe = code + "\n\n_tests = [([1,2,3,4,5], 3, 2), ([1,2,3,4,5], 6, -1), ([], 1, -1), ([1], 1, 0)]\n_passed = 0\nfor _arr, _t, _e in _tests:\n    try:\n        _f = binary_search if 'binary_search' in dir() else search if 'search' in dir() else binarySearch if 'binarySearch' in dir() else None\n        if _f and _f(_arr, _t) == _e: _passed += 1\n    except Exception: pass\nprint(_passed)"
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2:
                try:
                    passed = int(stdout2.strip())
                    score = passed / 4.0
                    return score >= 0.75, score, f"Code execution: {passed}/4 test cases passed"
                except Exception:
                    pass

    response_lower = response.lower()
    has_def = "def" in response_lower and ("binary" in response_lower or "search" in response_lower)
    has_mid = "mid" in response_lower or "middle" in response_lower
    has_loop = "while" in response_lower
    has_comparison = any(kw in response_lower for kw in ["<", ">", "==", "target"])
    score = 0.0
    if has_def: score += 0.25
    if has_mid: score += 0.25
    if has_loop: score += 0.25
    if has_comparison: score += 0.25
    return score >= 0.7, score, f"Heuristic: def={has_def}, mid={has_mid}, loop={has_loop}, comparison={has_comparison}"


def grade_json_parse(response: str) -> tuple[bool, float, str]:
    """Check if response correctly handles JSON parsing."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            probe = code + "\n\n_tests = [('{\"key\": \"value\"}', {'key': 'value'}), ('[1,2,3]', [1,2,3]), ('invalid', None)]\n_passed = 0\nfor _inp, _exp in _tests:\n    try:\n        _f = parse_json if 'parse_json' in dir() else parse if 'parse' in dir() else loads_json if 'loads_json' in dir() else None\n        if _f:\n            try:\n                _r = _f(_inp)\n                if _r == _exp: _passed += 1\n            except Exception:\n                if _exp is None: _passed += 1\n    except Exception: pass\nprint(_passed)"
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2:
                try:
                    passed = int(stdout2.strip())
                    score = passed / 3.0
                    return score >= 0.75, score, f"Code execution: {passed}/3 test cases passed"
                except Exception:
                    pass

    response_lower = response.lower()
    has_import = "import json" in response_lower
    has_loads = "json.loads" in response_lower or "json.load" in response_lower
    has_error_handling = "try" in response_lower and "except" in response_lower
    score = 0.0
    if has_import: score += 0.3
    if has_loads: score += 0.4
    if has_error_handling: score += 0.3
    return score >= 0.7, score, f"Heuristic: import={has_import}, loads={has_loads}, error_handling={has_error_handling}"


def grade_fizzbuzz(response: str) -> tuple[bool, float, str]:
    """Check if response contains correct FizzBuzz implementation."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            probe = code + "\n\n_r = fizzbuzz(15) if 'fizzbuzz' in dir() else fizz_buzz(15) if 'fizz_buzz' in dir() else None\nprint(_r)"
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2:
                expected = "[1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz', 11, 'Fizz', 13, 14, 'FizzBuzz']"
                if expected in stdout2:
                    return True, 1.0, "Perfect FizzBuzz implementation"

    response_lower = response.lower()
    has_fizz = "fizz" in response_lower
    has_buzz = "buzz" in response_lower
    has_loop = "for" in response_lower or "while" in response_lower
    has_modulo = "%" in response_lower
    has_15 = "15" in response_lower
    score = 0.0
    if has_fizz: score += 0.2
    if has_buzz: score += 0.2
    if has_loop: score += 0.2
    if has_modulo: score += 0.2
    if has_15: score += 0.2
    return score >= 0.7, score, f"Heuristic: fizz={has_fizz}, buzz={has_buzz}, loop={has_loop}, modulo={has_modulo}, 15={has_15}"


def grade_palindrome(response: str) -> tuple[bool, float, str]:
    """Check if response contains a correct palindrome checker."""
    code = extract_python_code(response)
    if code:
        ok, stdout, stderr = run_sandboxed(code)
        if ok:
            probe = code + "\n\n_tests = [('racecar', True), ('hello', False), ('a', True), ('', True), ('madam', True)]\n_passed = 0\nfor _inp, _exp in _tests:\n    try:\n        _f = is_palindrome if 'is_palindrome' in dir() else palindrome if 'palindrome' in dir() else check_palindrome if 'check_palindrome' in dir() else None\n        if _f and _f(_inp) == _exp: _passed += 1\n    except Exception: pass\nprint(_passed)"
            ok2, stdout2, _ = run_sandboxed(probe)
            if ok2:
                try:
                    passed = int(stdout2.strip())
                    score = passed / 5.0
                    return score >= 0.8, score, f"Code execution: {passed}/5 test cases passed"
                except Exception:
                    pass

    response_lower = response.lower()
    has_def = "def" in response_lower and ("palindrome" in response_lower or "palin" in response_lower)
    has_comparison = any(kw in response_lower for kw in ["[::-1]", "==", "reversed"])
    score = 0.0
    if has_def: score += 0.4
    if has_comparison: score += 0.6
    return score >= 0.7, score, f"Heuristic: def={has_def}, comparison={has_comparison}"


def grade_sql_query(response: str) -> tuple[bool, float, str]:
    """Check if response contains a correct SQL query."""
    response_lower = response.lower()
    has_select = "select" in response_lower
    has_from = "from" in response_lower
    has_join = "join" in response_lower
    where_count = response_lower.count("where")
    group_by = "group by" in response_lower
    has_aggregate = any(kw in response_lower for kw in ["count", "sum", "avg", "max", "min"])
    score = 0.0
    if has_select: score += 0.25
    if has_from: score += 0.25
    if has_join: score += 0.2
    if where_count >= 1: score += 0.15
    if group_by: score += 0.15
    return score >= 0.6, score, f"Heuristic: select={has_select}, from={has_from}, join={has_join}, where={where_count}, group_by={group_by}"


def grade_http_api(response: str) -> tuple[bool, float, str]:
    """Check if response contains correct HTTP API design."""
    response_lower = response.lower()
    has_app = any(kw in response_lower for kw in ["app", "api", "server"])
    has_route = any(kw in response_lower for kw in ["route", "@app", "endpoint", "path"])
    has_method = any(kw in response_lower for kw in ["get", "post", "put", "delete", "method"])
    has_json = any(kw in response_lower for kw in ["json", "response"])
    has_error = any(kw in response_lower for kw in ["error", "status", "404", "500"])
    score = 0.0
    if has_app: score += 0.2
    if has_route: score += 0.25
    if has_method: score += 0.25
    if has_json: score += 0.15
    if has_error: score += 0.15
    return score >= 0.6, score, f"Heuristic: app={has_app}, route={has_route}, method={has_method}, json={has_json}, error={has_error}"


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
TESTS = [
    {
        "name": "fibonacci",
        "category": "algorithms",
        "prompt": "Write a Python function `fib(n)` that returns the nth Fibonacci number. Use an iterative approach.",
        "grader": grade_fibonacci,
    },
    {
        "name": "reverse_string",
        "category": "strings",
        "prompt": "Write a Python function `reverse_string(s)` that reverses a string without using the `[::-1]` slice notation.",
        "grader": grade_reverse_string,
    },
    {
        "name": "binary_search",
        "category": "algorithms",
        "prompt": "Write a Python function `binary_search(arr, target)` that performs binary search on a sorted array. Return the index of target, or -1 if not found.",
        "grader": grade_binary_search,
    },
    {
        "name": "json_parse",
        "category": "data",
        "prompt": "Write a Python function `parse_json(s)` that parses a JSON string and returns the Python object. Handle errors gracefully by returning None.",
        "grader": grade_json_parse,
    },
    {
        "name": "fizzbuzz",
        "category": "classic",
        "prompt": "Write a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1 to n where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', and other numbers are strings.",
        "grader": grade_fizzbuzz,
    },
    {
        "name": "palindrome",
        "category": "strings",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome (reads the same forwards and backwards), False otherwise.",
        "grader": grade_palindrome,
    },
    {
        "name": "sql_query",
        "category": "sql",
        "prompt": "Write a SQL query that selects the name and total sales amount for the top 5 customers by total sales, from tables `customers` (id, name) and `orders` (id, customer_id, amount). Only include customers with more than 3 orders.",
        "grader": grade_sql_query,
    },
    {
        "name": "http_api",
        "category": "api",
        "prompt": "Design a simple REST API endpoint for a todo list application. Show the code for creating a new todo item with error handling.",
        "grader": grade_http_api,
    },
]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            test_name TEXT NOT NULL,
            category TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            correct INTEGER NOT NULL,
            score REAL NOT NULL,
            reasoning TEXT NOT NULL,
            created_at TEXT NOT NULL,
            response_time_s REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_accuracy_result(conn: sqlite3.Connection, result: AccuracyResult):
    conn.execute(
        """INSERT INTO accuracy_results
           (model, test_name, category, prompt, response, correct, score, reasoning, created_at, response_time_s)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (result.model, result.test_name, result.category, result.prompt,
         result.response, int(result.correct), result.score, result.reasoning,
         result.created_at, result.response_time_s),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def generate_response(model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> tuple[str, float]:
    """Generate a response from a model with retry logic."""
    start = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(API_RETRIES + 1):
        try:
            r = requests.post(
                f"{OLLAMA_API}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            r.raise_for_status()
            result = r.json()
            response = result.get("response", "")
            elapsed = time.perf_counter() - start
            return response, elapsed
        except Exception as e:
            last_error = e
            if attempt < API_RETRIES:
                time.sleep(1)

    elapsed = time.perf_counter() - start
    return f"ERROR: {last_error}", elapsed


def run_accuracy_evaluation(model: str, tests: list[dict] | None = None) -> list[AccuracyResult]:
    """Run all accuracy tests against a model."""
    tests = tests or TESTS
    results = []

    for test in tests:
        print(f"  [{model}] {test['name']}...", end=" ", flush=True)
        response, elapsed = generate_response(model, test["prompt"])
        correct, score, reasoning = test["grader"](response)
        print(f"score={score:.2f} ({'PASS' if correct else 'FAIL'})")

        result = AccuracyResult(
            model=model,
            test_name=test["name"],
            category=test["category"],
            prompt=test["prompt"],
            response=response,
            correct=correct,
            score=score,
            reasoning=reasoning,
            created_at=datetime.now(timezone.utc).isoformat(),
            response_time_s=round(elapsed, 2),
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_accuracy_report(conn: sqlite3.Connection):
    """Print accuracy results."""
    rows = conn.execute("""
        SELECT model, test_name, category, score, correct, reasoning, response_time_s
        FROM accuracy_results
        ORDER BY model, test_name
    """).fetchall()

    if not rows:
        print("No accuracy results found.")
        return

    print("\n" + "=" * 110)
    print("ACCURACY RESULTS")
    print("-" * 110)
    print(f"{'Model':<30} {'Test':<20} {'Category':<12} {'Score':>6} {'Result':>8} {'Time':>8}")
    print("-" * 110)
    for row in rows:
        status = "PASS" if row[4] else "FAIL"
        print(f"{row[0]:<30} {row[1]:<20} {row[2]:<12} {row[3]:>5.2f} {status:>8} {row[6]:>7.1f}s")
    print("=" * 110)


def print_model_accuracy_summary(conn: sqlite3.Connection):
    """Print average accuracy per model."""
    rows = conn.execute("""
        SELECT model,
               COUNT(*) as total_tests,
               SUM(correct) as passed,
               AVG(score) as avg_score,
               AVG(response_time_s) as avg_time
        FROM accuracy_results
        GROUP BY model
        ORDER BY avg_score DESC
    """).fetchall()

    if not rows:
        print("No accuracy results found.")
        return

    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("-" * 80)
    print(f"{'Model':<30} {'Tests':>6} {'Passed':>7} {'Avg Score':>10} {'Avg Time':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:<30} {row[1]:>6} {row[2]:>7} {row[3]:>9.2f} {row[4]:>9.1f}s")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Accuracy evaluation for coding models")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run accuracy evaluation")
    run.add_argument("model", help="Model name")
    run.add_argument("--test", action="append", help="Run specific test(s) only")

    rep = sub.add_parser("report", help="Show accuracy report")
    sum_parser = sub.add_parser("summary", help="Show accuracy summary")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    conn = init_db(DB_PATH)

    if args.command == "run":
        tests = TESTS
        if args.test:
            tests = [t for t in TESTS if t["name"] in args.test]
        results = run_accuracy_evaluation(args.model, tests)
        for r in results:
            save_accuracy_result(conn, r)
        print(f"\nSaved {len(results)} accuracy results to {DB_PATH}")

    elif args.command == "report":
        print_accuracy_report(conn)

    elif args.command == "summary":
        print_model_accuracy_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
