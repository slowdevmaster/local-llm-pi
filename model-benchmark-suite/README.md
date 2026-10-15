# Model Benchmark Suite

Benchmark and accuracy-evaluation tools for local GGUF models running on Ollama (Raspberry Pi 5).

## Install

```bash
pip install -r requirements.txt
```

## Usage

### benchmark.py — Latency, throughput, memory

```bash
# List models in Ollama
python benchmark.py list

# Download a model
python benchmark.py download llama3.1:8b-instruct

# Run standard benchmarks (short-qa, medium-reasoning, long-generation, coding)
python benchmark.py bench llama3.1:8b-instruct

# Custom prompt
python benchmark.py bench llama3.1:8b-instruct --prompt "Explain quantum computing"

# View report
python benchmark.py report

# Compare all models
python benchmark.py compare
```

### accuracy.py — Code generation quality

```bash
# Run all accuracy tests against a model
python accuracy.py run llama3.1:8b-instruct

# Run specific tests
python accuracy.py run llama3.1:8b-instruct --test fibonacci --test fizzbuzz

# View report
python accuracy.py report

# Summary by model
python accuracy.py summary
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API` | `http://localhost:11434` | Ollama API endpoint |
| `BENCHMARK_DB` | `~/.local/share/model-benchmark/results.db` | SQLite database path |

## Security

All model-generated code is executed in a sandboxed subprocess with:
- AST validation (blocks dangerous imports: os, subprocess, socket, etc.)
- CPU time limit (5 seconds)
- Memory limit (256 MB)
- File descriptor limit (32)
- Process limit (16)
- No network access (empty PATH)
