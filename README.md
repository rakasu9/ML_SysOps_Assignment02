# Distributed ResNet-50 Training on CIFAR-10

## Assignment 2: Implementation and Testing

This repository contains the implementation of distributed ResNet-50 training using Ring-AllReduce architecture for data-parallel deep learning.

---

## 📁 Project Structure

```
MLOPS/
├── 01_environment_setup.ipynb           # Environment setup and hardware verification
├── 02_baseline_resnet50_training.ipynb  # Single GPU/MPS baseline training
├── 03_distributed_training.ipynb        # Distributed training code structure
├── 04_testing_and_benchmarking.ipynb    # Testing and performance analysis
├── distributed_train.py                 # Standalone distributed training script
├── mlsysops.md                          # Assignment 1 submission (literature survey & design)
├── data/                                # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/                         # Model checkpoints
├── results/                             # Training results and plots
└── logs/                                # Training logs
```

---

## 🚀 Quick Start

### Step 1: Environment Setup

Run the environment setup notebook to verify your system and install dependencies:

```bash
jupyter notebook 01_environment_setup.ipynb
```

This will:
- Check hardware (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU)
- Install PyTorch, torchvision, and other dependencies
- Create necessary directories
- Save configuration

### Step 2: Baseline Training (Single GPU/MPS)

Run baseline training on a single device:

```bash
jupyter notebook 02_baseline_resnet50_training.ipynb
```

**Expected outputs:**
- Trained model checkpoint: `checkpoints/baseline_resnet50_cifar10.pth`
- Training history: `results/baseline_training_history.json`
- Performance metrics: `results/baseline_metrics.json`
- Training curves: `results/baseline_training_curves.png`

**Expected training time:** ~15-30 minutes on Apple M3 Pro MPS

### Step 3: Distributed Training (Multi-Process)

#### Option A: Run from Notebook (code structure only)
```bash
jupyter notebook 03_distributed_training.ipynb
```

#### Option B: Run distributed training script (actual multi-process)

For **2 processes** (simulating 2 GPUs):
```bash
torchrun --nproc_per_node=2 distributed_train.py
```

Or with custom parameters:
```bash
torchrun --nproc_per_node=2 distributed_train.py \
  --batch-size 64 \
  --epochs 10 \
  --lr 0.1 \
  --momentum 0.9
```

**Expected outputs:**
- Distributed model checkpoint: `checkpoints/distributed_resnet50.pth`
- Training history: `results/distributed_history.json`
- Performance metrics: `results/distributed_metrics.json`

**Note:** On Mac, this will simulate distributed training using CPU processes. For real multi-GPU training, use a system with multiple NVIDIA GPUs.

### Step 4: Testing and Benchmarking

Run comprehensive testing and performance analysis:

```bash
jupyter notebook 04_testing_and_benchmarking.ipynb
```

This notebook will:
- Load and evaluate trained models
- Generate per-class accuracy analysis
- Create confusion matrices
- Compare baseline vs distributed performance
- Calculate speedup and scaling efficiency
- Generate visualization plots

**Expected outputs:**
- Per-class accuracy plot: `results/per_class_accuracy.png`
- Confusion matrix: `results/confusion_matrix.png`
- Performance comparison: `results/performance_comparison.csv`
- Speedup analysis: `results/speedup_analysis.png`
- Comprehensive comparison: `results/baseline_vs_distributed_comparison.png`
- Test summary: `results/test_summary.json`

---

## 📊 Expected Results

### Performance Metrics

| Metric | Baseline (1 GPU) | Distributed (2 Processes) | Target |
|--------|------------------|---------------------------|--------|
| Training Time | ~15-30 min | ~8-15 min | <15 min |
| Speedup | 1.0x | ~1.8-2.0x | >1.5x |
| Scaling Efficiency | 100% | 90-100% | >80% |
| Final Accuracy | ~85-90% | ~85-90% | >80% |
| Throughput | ~500 img/s | ~900 img/s | >800 img/s |

### Accuracy Expectations

- **CIFAR-10 Validation Accuracy:** 85-90% after 10 epochs
- **Convergence:** Both baseline and distributed should achieve similar accuracy (within 1-2%)
- **Best performing classes:** ship, car, truck (~90%+)
- **Challenging classes:** cat, dog, bird (~75-85%)

---

## 🔧 Implementation Details

### Baseline Training
- **Model:** ResNet-50 (modified for CIFAR-10)
- **Dataset:** CIFAR-10 (50K train, 10K test)
- **Batch Size:** 128
- **Optimizer:** SGD with momentum (0.9)
- **Learning Rate:** 0.1 (with step decay)
- **Device:** MPS (Apple Silicon), CUDA (NVIDIA), or CPU

### Distributed Training
- **Framework:** PyTorch DistributedDataParallel (DDP)
- **Communication Backend:** Gloo (CPU/Mac), NCCL (NVIDIA GPUs)
- **Gradient Synchronization:** Ring-AllReduce (automatic via DDP)
- **Batch Size per Process:** 64
- **Effective Global Batch Size:** 128 (64 × 2 processes)
- **Learning Rate Scaling:** Linear scaling rule (LR × world_size)

### Ring-AllReduce Algorithm
PyTorch DDP automatically implements Ring-AllReduce for gradient synchronization:
1. Gradients computed locally on each process
2. Ring topology established between processes
3. Gradients exchanged in chunks with neighbors
4. After 2(N-1) steps, all processes have averaged gradients
5. **Communication Cost:** O(M) per process (constant!)

---

## 📈 Performance Analysis

### Communication Cost

For N processes and M parameters:

| Architecture | Per-Server Bandwidth | Scalability |
|--------------|---------------------|-------------|
| Parameter Server | O(M × N) | Poor (bottleneck) |
| **Ring-AllReduce** | **O(M)** | **Excellent** |

### Speedup Formula

$$\text{Speedup} = \frac{T_{\text{baseline}}}{T_{\text{distributed}}}$$

$$\text{Efficiency} = \frac{\text{Speedup}}{N} \times 100\%$$

Where N = number of processes

---

## 🧪 Testing Strategy

### 1. Correctness Testing
- ✓ Model loads successfully
- ✓ Predictions on test set
- ✓ Per-class accuracy analysis
- ✓ Confusion matrix visualization
- ✓ Classification report

### 2. Performance Testing
- ✓ Training time measurement
- ✓ Throughput calculation (images/second)
- ✓ Speedup computation
- ✓ Scaling efficiency analysis

### 3. Comparison Testing
- ✓ Baseline vs Distributed accuracy (should be within 1-2%)
- ✓ Training curves comparison
- ✓ Loss convergence verification
- ✓ Statistical validation

---

## 💻 System Requirements

### Minimum Requirements
- **OS:** macOS, Linux, or Windows
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **Python:** 3.8+
- **PyTorch:** 2.0+

### Recommended for Faster Training
- **GPU:** NVIDIA GPU with CUDA support OR Apple Silicon (M1/M2/M3)
- **RAM:** 16 GB+
- **Multi-GPU:** For true distributed training speedup

---

## 📚 References

1. **Assignment 1 Document:** `mlsysops.md` - Literature survey and design
2. **Papers Referenced:**
   - Dean et al. (2012) - Parameter Server framework
   - Goyal et al. (2017) - Linear scaling rule
   - Sergeev et al. (2018) - Ring-AllReduce in Horovod

---

## 🔍 Troubleshooting

### Issue: MPS not available on Mac
**Solution:** Update to PyTorch 2.0+ and macOS 12.3+

### Issue: Distributed training fails
**Solution:** Use `backend='gloo'` for CPU/Mac instead of `backend='nccl'`

### Issue: Out of memory
**Solution:** Reduce batch size in the notebooks/scripts

### Issue: Slow training
**Solution:** 
- Reduce `num_workers` if CPU bottleneck
- Reduce `batch_size` if memory bottleneck
- Use smaller number of epochs for testing

---

## 📝 Assignment Deliverables Checklist

- [x] **Code Implementation**
  - [x] Environment setup notebook
  - [x] Baseline training implementation
  - [x] Distributed training implementation
  - [x] Testing and benchmarking scripts

- [x] **Testing**
  - [x] Correctness verification
  - [x] Performance benchmarking
  - [x] Comparison analysis

- [ ] **Documentation** (To be completed)
  - [ ] Convert results to PDF report
  - [ ] Add team member contribution table
  - [ ] Upload code to GitHub
  - [ ] Include GitHub link in report

- [ ] **Report Sections** (To be written)
  - [ ] Abstract
  - [ ] Introduction
  - [ ] Revised Design (based on feedback)
  - [ ] Implementation Details
  - [ ] Results and Discussion
  - [ ] Performance Analysis
  - [ ] Conclusion
  - [ ] References

---

## 🎯 Next Steps

1. **Run all notebooks in sequence** (01 → 02 → 04)
2. **Run distributed training script** (optional, for comparison)
3. **Generate all visualizations** from notebook 04
4. **Compile results** into PDF report
5. **Upload code** to GitHub repository
6. **Write report** following assignment format

---

## 👥 Team Members

| Name | Roll Number | Contribution |
|------|-------------|--------------|
| [Your Name] | [Your Roll] | Implementation, Testing, Documentation |
| [Member 2] | [Roll 2] | [Contribution] |
| [Member 3] | [Roll 3] | [Contribution] |

**Corresponding Author (Submitter):** [Name in Bold]

---

## 📧 Contact

For questions or issues, contact: [your-email@domain.com]

---

**Last Updated:** February 13, 2026
