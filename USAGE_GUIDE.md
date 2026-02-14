# Complete Implementation - Usage Guide

## 📄 Single File Implementation

**File:** `complete_implementation.py` (All-in-One Script)

This single Python file contains the complete implementation of Assignment 2:
- Environment setup and verification
- Baseline training (single GPU/MPS)
- Distributed training (multi-process with Ring-AllReduce)
- Comprehensive testing and benchmarking
- Results visualization and analysis

---

## 🚀 How to Use

### 1. Install Dependencies
```bash
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn tqdm
```

### 2. Run Baseline Training
```bash
python complete_implementation.py --mode baseline --epochs 10 --batch-size 128
```

**Output:**
- Trained model: `checkpoints/baseline_resnet50_cifar10.pth`
- Training history: `results/baseline_training_history.json`
- Metrics: `results/baseline_metrics.json`
- Plots: `results/baseline_training_curves.png`

### 3. Run Testing & Analysis
```bash
python complete_implementation.py --mode test
```

**Output:**
- Per-class accuracy plot
- Confusion matrix
- Classification report
- Performance metrics

### 4. (Optional) Run Distributed Training
```bash
# For 2 processes (simulating 2 GPUs)
torchrun --nproc_per_node=2 complete_implementation.py --mode distributed --epochs 10 --batch-size 64

# For 4 processes
torchrun --nproc_per_node=4 complete_implementation.py --mode distributed --epochs 10 --batch-size 32
```

**Output:**
- Distributed model: `checkpoints/distributed_resnet50.pth`
- Training history: `results/distributed_history.json`
- Metrics: `results/distributed_metrics.json`
- Comparison plots (when testing is run again)

---

## 📋 Command Line Options

```
--mode {baseline,distributed,test,all}
    baseline     - Run single GPU/MPS training
    distributed  - Run distributed training (use with torchrun)
    test        - Run testing and benchmarking
    all         - Run baseline + testing

--batch-size INT      Batch size (default: 128)
--epochs INT          Number of epochs (default: 10)
--lr FLOAT           Learning rate (default: 0.1)
--momentum FLOAT     SGD momentum (default: 0.9)
--weight-decay FLOAT Weight decay (default: 1e-4)
--num-workers INT    Data loading workers (default: 2)
```

---

## 📊 Expected Results

### Performance Metrics (on M3 Pro Mac)
- **Training Time (Baseline):** ~15-30 minutes for 10 epochs
- **Validation Accuracy:** ~85-90%
- **Throughput:** ~500+ images/second
- **Model Size:** ~98MB (ResNet-50)

### With Distributed Training (2 processes)
- **Speedup:** ~1.8-2.0x
- **Scaling Efficiency:** >90%
- **Accuracy:** Similar to baseline (within 1-2%)

---

## 📁 Generated Files

After running, you'll have:

```
MLOPS/
├── complete_implementation.py    ← Single file with all code
│
├── checkpoints/
│   ├── baseline_resnet50_cifar10.pth
│   └── distributed_resnet50.pth
│
├── results/
│   ├── baseline_training_history.json
│   ├── baseline_metrics.json
│   ├── baseline_training_curves.png
│   ├── distributed_history.json
│   ├── distributed_metrics.json
│   ├── per_class_accuracy.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── performance_comparison.csv
│   └── baseline_vs_distributed_comparison.png
│
└── data/
    └── cifar-10-batches-py/  (auto-downloaded)
```

---

## 🎯 Quick Examples

### Example 1: Fast Testing (5 epochs)
```bash
python complete_implementation.py --mode baseline --epochs 5
python complete_implementation.py --mode test
```

### Example 2: Full Pipeline (10 epochs)
```bash
# Step 1: Baseline
python complete_implementation.py --mode baseline --epochs 10

# Step 2: Distributed (optional)
torchrun --nproc_per_node=2 complete_implementation.py --mode distributed --epochs 10 --batch-size 64

# Step 3: Analysis
python complete_implementation.py --mode test
```

### Example 3: Custom Settings
```bash
python complete_implementation.py \
  --mode baseline \
  --epochs 15 \
  --batch-size 256 \
  --lr 0.2 \
  --num-workers 4
```

---

## 🔍 Code Structure

The single file is organized into 8 sections:

1. **Environment Setup** - System check, device selection
2. **Data Loading** - CIFAR-10 loaders with distributed support
3. **Model Definition** - ResNet-50 adapted for CIFAR-10
4. **Baseline Training** - Single GPU/MPS training loop
5. **Distributed Training** - Multi-process DDP training
6. **Testing & Evaluation** - Model evaluation, metrics
7. **Visualization** - Plot generation functions
8. **Main Execution** - Command-line interface

---

## 🐛 Troubleshooting

**Problem:** `ImportError: No module named 'torch'`  
**Fix:** `pip install torch torchvision`

**Problem:** Out of memory  
**Fix:** Reduce `--batch-size` (try 64 or 32)

**Problem:** Distributed training fails  
**Fix:** Use `backend='gloo'` instead of `nccl` for CPU/Mac (already set in code)

**Problem:** Too slow  
**Fix:** Reduce `--epochs` for testing (use 3-5 instead of 10)

---

## 📝 For Your Report

### What This Code Demonstrates

✅ **[P0] Problem Formulation**
- Distributed data-parallel training
- Ring-AllReduce gradient synchronization
- Performance objectives (speedup, throughput)

✅ **[P1] Design**
- PyTorch DistributedDataParallel (DDP)
- Linear scaling rule for learning rate
- Synchronous SGD with momentum

✅ **[P2] Implementation**
- Complete working code in single file
- Baseline and distributed modes
- Automatic gradient synchronization (Ring-AllReduce via DDP)

✅ **[P3] Testing**
- Correctness verification (accuracy, per-class analysis)
- Performance benchmarking (speedup, efficiency)
- Comparative analysis with visualizations

---

## 📧 Quick Start (TL;DR)

```bash
# 1. Install
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn tqdm

# 2. Train
python complete_implementation.py --mode baseline --epochs 10

# 3. Test
python complete_implementation.py --mode test

# Done! Check results/ directory
```

---

## 🎓 For Assignment Submission

1. ✅ Code is ready in `complete_implementation.py`
2. ✅ Run the script to generate results
3. 📝 Write report sections using generated plots/metrics
4. 🌐 Upload to GitHub
5. 📋 Include GitHub link in report
6. 📄 Convert report to PDF

---

**File Created:** `complete_implementation.py` (1200+ lines, ~55KB)

**Contains:** All functionality from 4 notebooks + distributed training script merged into one file!

Good luck with your assignment! 🚀
