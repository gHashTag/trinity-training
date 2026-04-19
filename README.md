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

## 🌌 Ecosystem

Core dep: [zig-golden-float](https://github.com/gHashTag/zig-golden-float).

Cloud platforms:
- Railway — multi-account farm for distributed training
- Fly.io — multi-region swarm deployment

## 📜 License

MIT © gHashTag
```
