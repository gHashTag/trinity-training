# Trinity Training Infrastructure

HSLM (High-Speed Language Model) training infrastructure for Railway, Fly.io, and local CPU.

## Quick Start

### Local Training

\`\`\`bash
# Build HSLM trainer
zig build-exe src/hslm/train.zig -O ReleaseFast

# Run with TinyStories dataset
docker build -f deploy/Dockerfile.hslm-train-local.v5 -t hslm-train .
docker run --rm -v $(pwd)/data:/data hslm-train
\`\`\`

### Railway Deployment

\`\`\`bash
# Deploy to Railway
./deploy/scripts/railway_deploy_experiment.sh r4

# Environment variables
HSLM_OPTIMIZER=adamw HSLM_LR=3e-4 HSLM_BATCH=66 HSLM_STEPS=100000
\`\`\`

### Fly.io Deployment

\`\`\`bash
# Deploy to Fly.io
fly deploy -c fly.toml

# Or launch multi-region swarm
./deploy/launch-swarm.sh fly
\`\`\`

## HSLM Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| HSLM_FRESH | New training (1) or resume (0) | 1 |
| HSLM_OPTIMIZER | adam, adamw, lamb | adamw |
| HSLM_LR | Learning rate | 3e-4 |
| HSLM_LR_SCHEDULE | sacred, cosine-restarts | sacred |
| HSLM_BATCH | Batch size | 66 |
| HSLM_GRAD_ACCUM | Gradient accumulation | 1 |
| HSLM_CONTEXT | Context window | 81 |
| HSLM_STEPS | Total training steps | 100000 |
| HSLM_WD | Weight decay | 0.1 |
| HSLM_DROPOUT | Dropout rate | 0.1 |
| HSLM_PROFILE | Training profile | s3-multiobj |

## Directory Structure

\`\`\`
src/
├── hslm/           # Core HSLM implementation
├── cli/             # Training CLI entrypoints
└── tri/             # Training orchestration
deploy/
├── Dockerfile.*      # Docker images
├── prebuilt/        # Pre-compiled binaries
└── scripts/         # Deployment scripts
tools/
├── railway_deploy.zig
└── railway_cleanup.py
specs/
└── tri/
    └── train_types.tri  # Training type specs
\`\`\`

## Training Objectives

- **NTG** (Next Token Generation): Standard language modeling
- **JEPA** (Jigsaw Predictive Coding): Self-supervised masked prediction
- **NCA** (Neural Cellular Automata): Spatial reasoning pre-training
- **Hybrid**: Combined objectives (NTP 50% + JEPA 25% + NCA 25%)

## Model Architecture

- Ternary Neural Networks (-1, 0, +1 weights)
- ~1.24M total parameters, compressed to ~248KB
- Trinity blocks with Sacred attention
- STE (Straight-Through Estimator) for true ternary gradients

## License

MIT
