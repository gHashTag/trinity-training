# Trinity Training Infrastructure

> HSLM (High-Speed Language Model) training infrastructure for Railway, Fly.io, and local CPU training.

[![Zig](https://img.shields.io/badge/Zig-0.15.2-orange?logo=zig)](https://ziglang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What is Trinity Training?

Pure Zig implementation of ternary neural network training infrastructure. Optimized for:

- **Railway** - Cloud GPU/CPU training with persistent storage
- **Fly.io** - Multi-region deployment with automatic scaling
- **Local CPU** - Development and experimentation on standard hardware

## Quick Start

### Local Training

```bash
# Build HSLM trainer
zig build-exe src/hslm/train.zig -O ReleaseFast

# Run with TinyStories dataset
docker build -f local/Dockerfile.hslm-train-local.v5 -t hslm-train .
docker run --rm -v $(pwd)/data:/data hslm-train
```

### Railway Deployment

```bash
# Deploy experiment r4 (AdamW + Sacred LR)
./railway/railway_deploy_experiment.sh r4

# Custom configuration
HSLM_OPTIMIZER=adamw HSLM_LR=3e-4 HSLM_BATCH=66 HSLM_STEPS=100000
```

### Fly.io Deployment

```bash
# Deploy to Fly.io (Singapore region)
fly deploy -c flyio/fly.toml

# Multi-region swarm (5 regions: iad, lax, ams, sin, nrt)
./launch-swarm.sh fly
```

## Training Profiles

| Profile | Optimizer | LR | Context | Objective |
|---------|-----------|-----|---------|-----------|
| r4 | adamw | 3e-4 (sacred) | s3-multiobj |
| r5 | lamb | 1e-3 | s3-multiobj |
| r6 | adamw | 5e-4 (cosine-restarts) | s3-multiobj |
| r7 | lamb | 1e-3 | s3-multiobj |

## HSLM Environment Variables

```bash
HSLM_FRESH=1          # New training (1) or resume from checkpoint (0)
HSLM_OPTIMIZER=adamw    # adam, adamw, lamb
HSLM_LR=3e-4            # Learning rate
HSLM_LR_SCHEDULE=sacred  # sacred, cosine-restarts
HSLM_BATCH=66           # Batch size
HSLM_GRAD_ACCUM=1       # Gradient accumulation steps
HSLM_CONTEXT=81          # Context window size
HSLM_STEPS=100000        # Total training steps
HSLM_WD=0.1            # Weight decay
HSLM_DROPOUT=0.1        # Dropout rate
HSLM_PROFILE=s3-multiobj  # Training profile
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trinity Training Infrastructure               │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │    HSLM     │  │   Dockerfiles │  │   Scripts   │  │
│  │  (100+ files)│  │    Railway     │  │  │  │
│  │  └─────────────┘  │    Fly.io      │  │  │  │
│  │                   │  │    Local      │  │  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

- **src/hslm/** - Complete HSLM implementation
  - `model.zig` - Ternary neural network (~1.24M params)
  - `trainer.zig` - Full trainer with autograd + STE + AdamW
  - `train.zig` - Training loop (zeroth-order optimization)
  - `tjepa_trainer.zig` - T-JEPA trainer with dual optimizers
  - `bpe_train.zig` - BPE tokenizer (469 merge rules)
  - `autograd.zig` - Reverse-mode automatic differentiation
  - `data.zig` - Streaming data loader with batching

- **src/cli/** - Training entrypoints
  - `entrypoint_train.zig` - Zig replacement for bash entrypoint
  - Environment-based configuration (steps, LR, batch size, etc.)
  - Health server for Railway compliance
  - Automatic checkpoint resumption

- **src/tri/** - Training orchestration
  - `local_farm.zig` - Multi-worker local training (up to 48 workers)
  - Crash tracking and PPL monitoring

### Deployment

- **deploy/** - Docker images and scripts
  - `Dockerfile.hslm-train` - Base image for Railway/Fly.io
  - `Dockerfile.hslm-train-local.v5` - Production local image
  - `Dockerfile.hslm-train-local.arm64` - Apple Silicon optimized
  - `prebuilt/v2/** - Cross-compiled binaries for deployment

- **railway/** - Railway configuration
  - `railway.json` - Main configuration
  - `railway_deploy_experiment.sh` - Experiment deployment script

- **flyio/** - Fly.io configuration
  - `fly.toml` - Main configuration
  - `fly.swarm.toml` - Multi-region swarm config
  - `deploy-flyio.sh` - Deployment script
  - `benchmark_flyio.sh` - Performance benchmarking

- **tools/** - Utility scripts
  - `railway_deploy.zig` - Zig-based Railway deployment
  - `railway_cleanup.py` - Service cleanup for multiple accounts

## Training Objectives

### NTG (Next Token Generation)
Standard autoregressive language modeling. Predicts the next token given context.

### JEPA (Jigsaw Predictive Coding)
Self-supervised masked prediction. Predicts missing spatial patches from context.

### NCA (Neural Cellular Automata)
Spatial reasoning pre-training. Learns local update rules from data.

### Hybrid
Combined objectives (NTG 50% + JEPA 25% + NCA 25%).

## Model Architecture

- **Ternary Weights**: {-1, 0, +1} instead of floats
- **Compression**: ~1.24M parameters → ~248KB (13,000×)
- **Trinity Blocks**: Sacred φ-geometry attention mechanism
- **STE**: Straight-Through Estimator for true ternary gradients
- **Zeroth-Order**: Perturb-and-measure optimization (no backprop required)

## Model Configuration

```zig
const VOCAB_SIZE = 729;
const EMBED_DIM = 243;
const HIDDEN_DIM = 729;
const CONTEXT_LEN = 81;
const NUM_BLOCKS = 3;
```

## Training Configuration

- **Learning Rate**: 3e-4 peak with φ-cosine decay to 1e-6
- **Warmup**: 5,000 steps
- **Batch Size**: 64-256 tokens
- **Gradient Clipping**: 1.0 (BitNet-style)
- **Weight Decay**: 0.1

## Dataset

**TinyStories**: 100k stories from HuggingFace. Automatically downloaded on first run.

```bash
# Prepare dataset manually
python3 - <<EOF
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories-all")
ds["train"].to_json("data/tinystories/train_100k.json")
EOF
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related

- [Trinity](https://github.com/gHashTag/trinity) - Core framework
- [zig-hslm](https://codeberg.org/gHashTag/zig-hslm) - Official HSLM numerical library
