# 🚀 QUICK START GUIDE

## Assignment 2: Distributed ResNet-50 Training Implementation

---

## ⚡ Fast Track (3 Steps)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Baseline Training
Open and run:
```bash
jupyter notebook 02_baseline_resnet50_training.ipynb
```
⏱️ Time: ~15-30 minutes

### 3️⃣ Run Testing & Analysis
Open and run:
```bash
jupyter notebook 04_testing_and_benchmarking.ipynb
```
⏱️ Time: ~5 minutes

---

## 📊 What You Get

After running the above, you'll have:

✅ **Trained Model**
- `checkpoints/baseline_resnet50_cifar10.pth`

✅ **Performance Metrics**
- Training accuracy: ~85-90%
- Validation accuracy: ~85-90%
- Training time logged

✅ **Visualizations**
- Training curves (loss, accuracy)
- Per-class accuracy analysis
- Confusion matrix
- Performance plots

✅ **Results Files**
- `results/baseline_training_history.json`
- `results/baseline_metrics.json`
- `results/baseline_training_curves.png`
- `results/per_class_accuracy.png`
- `results/confusion_matrix.png`

---

## 🔄 Optional: Distributed Training

For comparison with multi-process training:

```bash
torchrun --nproc_per_node=2 distributed_train.py
```

Then re-run notebook 04 to compare results.

---

## 📝 Files Overview

| File | Purpose | Run Order |
|------|---------|-----------|
| **01_environment_setup.ipynb** | Verify system, install deps | 1st (optional) |
| **02_baseline_resnet50_training.ipynb** | Train baseline model | 2nd (required) |
| **03_distributed_training.ipynb** | Distributed code explanation | Reference only |
| **04_testing_and_benchmarking.ipynb** | Test and analyze results | 3rd (required) |
| **distributed_train.py** | Distributed training script | Optional |
| **README.md** | Full documentation | Reference |

---

## 🎯 For Assignment Submission

### What's Implemented
✅ [A1] Literature Survey → See `mlsysops.md`  
✅ [A2] Problem Formulation → See `mlsysops.md`  
✅ [A3] Initial Design → See `mlsysops.md`  
✅ [P1] Revised Design → In code comments  
✅ [P2] Implementation → All `.ipynb` files + `distributed_train.py`  
✅ [P3] Testing → `04_testing_and_benchmarking.ipynb`  

### Next Steps for Report
1. ✍️ Write implementation details section
2. 📊 Add results from `results/` folder
3. 🔍 Add discussion on performance/deviations
4. 📚 Update references
5. 🎨 Convert to PDF
6. 🌐 Upload code to GitHub
7. 📋 Add team member table with contributions

---

## ⚠️ Quick Troubleshooting

**Problem:** Jupyter not installed  
**Fix:** `pip install jupyter notebook`

**Problem:** PyTorch not found  
**Fix:** `pip install torch torchvision`

**Problem:** Out of memory  
**Fix:** In notebook 02, reduce `BATCH_SIZE` from 128 to 64

**Problem:** Training too slow  
**Fix:** Reduce `NUM_EPOCHS` from 10 to 5 for testing

**Problem:** Distributed training fails  
**Fix:** This is optional - you can complete the assignment with baseline only

---

## 📧 Questions?

Check **README.md** for detailed documentation!

---

**Ready to start?** 

Run: `jupyter notebook 02_baseline_resnet50_training.ipynb`

Good luck! 🎓
