# Assignment 2: ResNet-50 CIFAR-10 Training Results (10 Epochs)

**Date:** February 13, 2026  
**Model:** ResNet-50 (23,520,842 parameters)  
**Dataset:** CIFAR-10 (50,000 train, 10,000 test)  
**Hardware:** Apple M3 Pro (MPS acceleration)

---

## 📊 Final Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | **62.58%** |
| **Best Validation Accuracy** | **62.76%** (Epoch 9) |
| **Final Training Accuracy** | 62.48% |
| **Total Training Time** | 41.16 minutes (2,469.67 seconds) |
| **Average Time per Epoch** | 218.53 seconds (~3.6 min) |
| **Throughput** | 202.46 images/second |

---

## 📈 Training Progression

### Epoch-by-Epoch Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time (s) | Learning Rate |
|-------|-----------|-----------|----------|---------|----------|---------------|
| 1     | 3.0047    | 14.33%    | 2.0868   | 22.10%  | 217.31   | 0.1          |
| 2     | 1.9559    | 25.25%    | 1.8439   | 30.59%  | 219.94   | 0.1          |
| 3     | 1.7771    | 33.61%    | 1.6005   | 40.07%  | 215.65   | 0.1          |
| 4     | 1.5900    | 40.91%    | 1.5696   | 43.75%  | 218.40   | 0.1          |
| 5     | 1.4218    | 47.64%    | 1.3274   | 51.51%  | 217.74   | 0.1          |
| 6     | 1.2235    | 55.29%    | 1.1582   | 57.60%  | 218.55   | 0.01         |
| 7     | 1.1641    | 57.58%    | 1.1280   | 59.21%  | 218.49   | 0.01         |
| 8     | 1.1074    | 59.78%    | 1.0717   | 61.42%  | 224.72   | 0.01         |
| 9     | 1.0495    | 62.02%    | 1.0360   | **62.76%** | 218.03   | 0.001        |
| 10    | 1.0401    | 62.48%    | 1.0388   | 62.58%  | 216.50   | 0.001        |

**Key Observations:**
- Learning rate schedule: 0.1 (epochs 1-5) → 0.01 (epochs 6-8) → 0.001 (epochs 9-10)
- Largest accuracy jump at epoch 3: +9.48% validation accuracy
- Model converged after epoch 9 (best validation accuracy)
- Minimal overfitting: train accuracy (62.48%) ≈ val accuracy (62.58%)

---

## 🎯 Per-Class Performance

### Class Accuracy Rankings

| Rank | Class  | Accuracy | Precision | Recall | F1-Score |
|------|--------|----------|-----------|--------|----------|
| 1    | **Car**    | **75.70%** | 0.827 | 0.757 | 0.791 |
| 2    | **Truck**  | **75.90%** | 0.746 | 0.759 | 0.752 |
| 3    | **Ship**   | **75.90%** | 0.713 | 0.759 | 0.735 |
| 4    | **Frog**   | **73.60%** | 0.711 | 0.736 | 0.723 |
| 5    | Horse  | 65.40% | 0.712 | 0.654 | 0.682 |
| 6    | Plane  | 63.90% | 0.635 | 0.639 | 0.637 |
| 7    | Dog    | 58.80% | 0.492 | 0.588 | 0.536 |
| 8    | Bird   | 54.10% | 0.461 | 0.541 | 0.498 |
| 9    | Deer   | 47.10% | 0.558 | 0.471 | 0.511 |
| 10   | Cat    | 35.40% | 0.427 | 0.354 | 0.387 |

### Performance Categories

**🟢 Strong Performance (>70%):** Car, Truck, Ship, Frog  
**🟡 Moderate Performance (50-70%):** Horse, Plane, Dog, Bird  
**🔴 Weak Performance (<50%):** Deer, Cat

---

## 🔬 Model Analysis

### Confusion Matrix Insights

The model shows:
- **Best recognition:** Vehicles (car, truck, ship) - clear shapes and distinctive features
- **Good recognition:** Frog - distinctive color and shape patterns
- **Moderate recognition:** Bird, dog, horse - animal complexity
- **Challenging:** Cat vs Dog confusion - similar features
- **Most difficult:** Cat - confused with dog (similar animals)

### Training Characteristics

**Convergence Pattern:**
```
Initial rapid learning (Epochs 1-5):
  - Accuracy improved from 22.10% → 51.51% (+29.41%)
  - Loss decreased from 2.09 → 1.33 (-36.4%)

Steady refinement (Epochs 6-8):
  - Accuracy improved from 57.60% → 61.42% (+3.82%)
  - Learning rate reduced to 0.01

Fine-tuning (Epochs 9-10):
  - Accuracy improved from 62.76% → 62.58% (converged)
  - Learning rate reduced to 0.001
```

---

## 💾 Generated Artifacts

### Checkpoints
- `checkpoints/baseline_resnet50_cifar10.pth` (180 MB)
  - Trained model weights
  - Optimizer state
  - Training history
  - Configuration

### Results Files
1. **baseline_metrics.json** - Summary statistics
2. **baseline_training_history.json** - Complete epoch-by-epoch data
3. **classification_report.txt** - Detailed per-class metrics
4. **baseline_training_curves.png** - Training/validation curves (4 plots)
5. **per_class_accuracy.png** - Bar chart of class accuracies
6. **confusion_matrix.png** - Confusion matrices (raw + normalized)

---

## 🚀 Hyperparameters Used

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-50 (modified for CIFAR-10) |
| Optimizer | SGD with Momentum |
| Initial Learning Rate | 0.1 |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| Batch Size | 128 |
| LR Schedule | MultiStepLR (milestones: [5, 8], gamma: 0.1) |
| Data Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalization | Mean=[0.4914, 0.4822, 0.4465], Std=[0.2470, 0.2435, 0.2616] |

---

## 📝 Next Steps for Assignment

### 1. Distributed Training Comparison
Run multi-GPU distributed training to compare speedup:
```bash
torchrun --nproc_per_node=2 complete_implementation.py --mode distributed --epochs 10 --batch-size 64
```

### 2. Report Generation
Include in PDF report:
- ✅ Training curves (see `baseline_training_curves.png`)
- ✅ Confusion matrix analysis (see `confusion_matrix.png`)
- ✅ Per-class accuracy breakdown (see `per_class_accuracy.png`)
- ✅ Performance metrics table (see above)
- ✅ Convergence analysis (learning rate schedule impact)
- ⏳ Distributed vs baseline speedup comparison
- ⏳ Scalability analysis (efficiency metrics)

### 3. Code Repository
Upload to GitHub:
- `complete_implementation.py` - Main implementation
- `distributed_train.py` - Standalone distributed script
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `QUICKSTART.md` - Quick start guide
- `USAGE_GUIDE.md` - Detailed usage

### 4. Assignment Deliverables
- [ ] PDF Report with all sections completed
- [ ] GitHub repository link
- [ ] Results visualization in report
- [ ] Performance analysis and discussion
- [ ] Team member contributions documented

---

## 📊 Statistical Summary

### Overall Model Performance
- **Macro Average Precision:** 0.628
- **Macro Average Recall:** 0.626
- **Macro Average F1-Score:** 0.625
- **Overall Accuracy:** 62.58%

### Training Efficiency
- **Training samples processed:** 500,000 (50,000 × 10 epochs)
- **Validation samples evaluated:** 100,000 (10,000 × 10 epochs)
- **Total forward+backward passes:** 3,910 batches

### Hardware Utilization
- **Device:** Apple M3 Pro (MPS backend)
- **Memory usage:** ~180 MB model + data buffers
- **Batch processing:** 128 images per iteration
- **Throughput achieved:** 202.46 images/second

---

## 🎓 Comparison with 3-Epoch Results

| Metric | 3 Epochs | 10 Epochs | Improvement |
|--------|----------|-----------|-------------|
| Validation Accuracy | 23.07% | **62.58%** | **+39.51%** |
| Training Time | 114 min | 41 min | Optimized |
| Ship Class Accuracy | 71.60% | **75.90%** | +4.30% |
| Frog Class Accuracy | 57.30% | **73.60%** | +16.30% |
| Cat Class Accuracy | 18.80% | **35.40%** | +16.60% |

**Key Improvement:** Extended training significantly improved model generalization and achieved 2.7× better accuracy.

---

## ✅ Conclusion

The ResNet-50 model achieved **62.58% validation accuracy** on CIFAR-10 after 10 epochs of training, demonstrating:

1. **Effective Learning:** Smooth convergence with proper learning rate scheduling
2. **Balanced Performance:** Minimal overfitting (train ≈ validation accuracy)
3. **Class Variation:** Strong performance on vehicles (75%+), moderate on animals (35-65%)
4. **Efficient Training:** 41 minutes on Apple M3 Pro @ 202 images/second

The model is ready for distributed training comparison to evaluate scalability and speedup metrics for the final assignment report.

---

**Generated:** February 13, 2026  
**Script:** `complete_implementation.py`  
**Mode:** Baseline (Single MPS GPU)
