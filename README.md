# Local LLM on Raspberry Pi 5

Complete toolkit for running, benchmarking, and evaluating local GGUF models on a Raspberry Pi 5 with Ollama.

## Repository Structure

```
├── scripts/
│   ├── ollama-run.sh          # Cron @reboot — starts Ollama if not running
│   └── metrics.sh             # Cron every-minute — logs CPU/RAM/disk usage
├── model-benchmark-suite/
│   ├── benchmark.py           # Latency, throughput, memory benchmarks
│   ├── accuracy.py            # Code generation accuracy evaluation
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Benchmark suite docs
└── modelfiles/
    ├── llama3.1-8b-instruct/Modelfile
    ├── deepseek-coder-6.7b-instruct/Modelfile
    ├── qwen2.5-coder-1.5b-instruct/Modelfile
    └── qwen2.5-coder-3b-instruct/Modelfile
```

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Set up boot persistence

```bash
# Copy the boot script
cp scripts/ollama-run.sh /home/jimmy/ollama_root/ollama-run.sh
chmod +x /home/jimmy/ollama_root/ollama-run.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "@reboot /home/jimmy/ollama_root/ollama-run.sh") | crontab -
```

### 3. Import models

```bash
# Copy Modelfiles to your gguf directory, then:
ollama create llama3.1-8b-instruct -f /path/to/Modelfile
```

### 4. Run benchmarks

```bash
cd model-benchmark-suite
pip install -r requirements.txt
python benchmark.py bench llama3.1-8b-instruct
python accuracy.py run llama3.1-8b-instruct
```

## Hardware

- Raspberry Pi 5 (4-core ARM Cortex-A76, 8 GB RAM)
- CPU-only (no CUDA)
- Models: Q4_K_M quantized GGUF

## License

MIT
