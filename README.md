# trinity-training

[![Zig](https://img.shields.io/badge/Zig-0.15+-F7A41D?logo=zig&logoColor=white)](https://ziglang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HSLM](https://img.shields.io/badge/HSLM-ternary-purple)](https://arxiv.org/abs/2306.13931)
[![Ecosystem](https://img.shields.io/badge/Trinity-Training-green)](https://github.com/gHashTag/trinity-training)

> **HSLM (Hybrid Symbolic Language Model) training infrastructure** — Ternary neural networks, Beal conjecture, zeroth-order optimization, Railway deployment.

## ✨ Features

- 🔢 **HSLM Model** — ~1.24M ternary parameters, ~248KB compressed
- 🧠 **Sacred Attention** — φ-weighted mechanism for HSLM
- 📐 **Autograd** — reverse-mode automatic differentiation
- 🤖 **Zeroth-Order** — perturb-and-measure optimization (no backprop)
- 🚂 **T-JEPA** — jigsaw predictive coding self-supervision
- 🌐 **Railway Deployment** — cloud farm for distributed training
- 📊 **Benchmarks** — MNIST, CIFAR-10, neural network tests

## 📦 Installation

```bash
# Clone with zig-golden-float submodule
git clone --recursive https://github.com/gHashTag/trinity-training.git
cd trinity-training
git submodule update --init --recursive
```

## 🏗️ Modules

```
src/
├── hslm/          (70+ files)
│   ├── model.zig
│   ├── trainer.zig
│   ├── train.zig
│   ├── autograd.zig
│   ├── attention.zig
│   ├── sacred_attention.zig
│   └── ...
├── bench/          benchmarks
├── data_loaders/  MNIST, CIFAR-10
└── tri/             training orchestration
data/               (208MB)
```

## 🌌 Trinity Ecosystem

> Golden Ratio mathematics meets computational physics and AI.

| Repository | Purpose | Status |
|---|---|---|
| [trinity](https://github.com/gHashTag/trinity) | 🎯 Orchestrator, agents, API, MCP server | ✅ Main |
| [zig-golden-float](https://github.com/gHashTag/zig-golden-float) | 🔢 Numeric core: GF16, TF3, VSA, JIT | [![CI](https://img.shields.io/github/actions/workflow/status/gHashTag/zig-golden-float/ci.yml?branch=main)](https://github.com/gHashTag/zig-golden-float/actions) |
| [trinity-training](https://github.com/gHashTag/trinity-training) | 🧠 ML: HSLM, benchmarks, datasets | ✅ Here |
| [t27](https://github.com/gHashTag/t27) | 📜 Ternary SSOT + Rust bootstrap | 📜 Language |
| [vibee-lang](https://github.com/gHashTag/vibee-lang) | 🎵 VIBEE language spec (.tri/.vibee) | 📜 Language |
| [zig-hdc](https://github.com/gHashTag/zig-hdc) | 🧩 Hyperdimensional: VSA, HRR | ✅ |
| [zig-sacred-geometry](https://github.com/gHashTag/zig-sacred-geometry) | 📐 Sacred φ-geometry, Beal | ✅ |
| [zig-physics](https://github.com/gHashTag/zig-physics) | ⚛️ Quantum: QCD, gravity, dark matter | ✅ |
| [zig-knowledge-graph](https://github.com/gHashTag/zig-knowledge-graph) | 🕸️ KG server + CLI | ✅ |
| [zig-agents](https://github.com/gHashTag/zig-agents) | 🤖 Agents: MCP, autonomous | ✅ |
| [zig-crypto-mining](https://github.com/gHashTag/zig-crypto-mining) | 💰 BTC mining + DePIN | ✅ |
| [trinity-fpga](https://github.com/gHashTag/trinity-fpga) | 🔌 FPGA: Verilog synthesis | 🔄 WIP |

**Cloud Platforms:**
- Railway — multi-account farm for distributed training
- Fly.io — multi-region swarm deployment

## 📜 License

MIT © gHashTag
