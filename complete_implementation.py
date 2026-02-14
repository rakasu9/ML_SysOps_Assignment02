#!/usr/bin/env python3
"""
Assignment 2: Distributed ResNet-50 Training on CIFAR-10
Complete Implementation - All-in-One Script

This script contains all functionality:
1. Environment setup and verification
2. Baseline single-GPU/MPS training
3. Distributed multi-process training
4. Testing and benchmarking
5. Results visualization and analysis

Usage:
  # Run baseline training
  python complete_implementation.py --mode baseline --epochs 10 --batch-size 128
  
  # Run distributed training (2 processes)
  torchrun --nproc_per_node=2 complete_implementation.py --mode distributed --epochs 10 --batch-size 64
  
  # Run testing and analysis
  python complete_implementation.py --mode test
  
  # Run everything sequentially
  python complete_implementation.py --mode all
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import os
import sys
import time
import json
import argparse
import platform
from datetime import datetime
from tqdm import tqdm

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# ============================================================================
# SECTION 1: ENVIRONMENT SETUP AND UTILITIES
# ============================================================================

def check_environment():
    """Check and display system environment"""
    print("=" * 70)
    print("SYSTEM ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print("=" * 70)
    
    # Check hardware acceleration
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    print("\nHARDWARE ACCELERATION:")
    print(f"  CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    
    print(f"  MPS (Apple Silicon) Available: {mps_available}")
    
    # Determine device
    if cuda_available:
        device = torch.device("cuda")
        device_name = "CUDA"
    elif mps_available:
        device = torch.device("mps")
        device_name = "MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"\n  Selected Device: {device_name}")
    print("=" * 70)
    
    return device, device_name


def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs', 'results', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(f"✓ Created directories: {', '.join(directories)}")


# ============================================================================
# SECTION 2: DATA LOADING
# ============================================================================

def get_cifar10_loaders(batch_size, num_workers=2, distributed=False, rank=0, world_size=1):
    """
    Create CIFAR-10 data loaders
    
    Args:
        batch_size: Batch size (per process for distributed)
        num_workers: Number of data loading workers
        distributed: Whether to use distributed sampler
        rank: Process rank (for distributed)
        world_size: Total number of processes
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    if distributed:
        train_sampler = DistributedSampler(
            trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        test_sampler = DistributedSampler(
            testset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle_train = True
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle_train,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader, train_sampler


# ============================================================================
# SECTION 3: MODEL DEFINITION
# ============================================================================

def create_resnet50(num_classes=10, device=None):
    """Create ResNet-50 model adapted for CIFAR-10"""
    model = torchvision.models.resnet50(weights=None)
    
    # Modify for CIFAR-10 (32x32 images instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if device is not None:
        model = model.to(device)
    
    return model


# ============================================================================
# SECTION 4: BASELINE TRAINING (SINGLE GPU/MPS)
# ============================================================================

def train_epoch_baseline(model, trainloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch (baseline)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    epoch_start = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_time = time.time() - epoch_start
    
    return {
        'loss': running_loss / len(trainloader),
        'accuracy': 100. * correct / total,
        'time': epoch_time
    }


def validate_baseline(model, testloader, criterion, device):
    """Validate the model (baseline)"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc='Validating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'loss': test_loss / len(testloader),
        'accuracy': 100. * correct / total
    }


def run_baseline_training(args):
    """Run baseline single-GPU/MPS training"""
    print("\n" + "=" * 70)
    print("BASELINE TRAINING (SINGLE GPU/MPS)")
    print("=" * 70)
    
    device, device_name = check_environment()
    create_directories()
    
    print(f"\nConfiguration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {device_name}")
    
    # Data
    trainloader, testloader, _ = get_cifar10_loaders(
        args.batch_size, args.num_workers, distributed=False
    )
    print(f"\nDataset: CIFAR-10")
    print(f"  Training samples: {len(trainloader.dataset)}")
    print(f"  Test samples: {len(testloader.dataset)}")
    
    # Model
    model = create_resnet50(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet-50")
    print(f"  Parameters: {total_params:,}")
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 8], gamma=0.1
    )
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': [], 'learning_rate': []
    }
    
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70 + "\n")
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch_baseline(
            model, trainloader, criterion, optimizer, device, epoch, args.epochs
        )
        val_metrics = validate_baseline(model, testloader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['epoch_time'].append(train_metrics['time'])
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Epoch Time: {train_metrics['time']:.2f}s | LR: {current_lr}")
        print("-" * 70)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print("=" * 70)
    
    # Save results
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'device': str(device),
            'total_time': total_time
        }
    }
    
    torch.save(checkpoint, 'checkpoints/baseline_resnet50_cifar10.pth')
    
    with open('results/baseline_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Metrics
    total_samples = len(trainloader.dataset) * args.epochs
    throughput = total_samples / total_time
    
    metrics = {
        'device': device_name,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'total_training_time_seconds': total_time,
        'average_epoch_time_seconds': np.mean(history['epoch_time']),
        'throughput_images_per_second': throughput,
        'best_val_accuracy': max(history['val_acc']),
        'final_val_accuracy': history['val_acc'][-1],
        'final_train_accuracy': history['train_acc'][-1]
    }
    
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Checkpoint saved: checkpoints/baseline_resnet50_cifar10.pth")
    print(f"✓ Results saved: results/baseline_*.json")
    
    # Generate plots
    generate_training_plots(history, 'baseline')
    
    return model, history, metrics


# ============================================================================
# SECTION 5: DISTRIBUTED TRAINING
# ============================================================================

def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend='gloo')  # Use 'nccl' for NVIDIA GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def get_device_distributed(rank):
    """Get device for distributed training"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def train_epoch_distributed(model, trainloader, criterion, optimizer, device, rank, epoch, total_epochs):
    """Train for one epoch (distributed)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # DDP handles gradient synchronization
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_time = time.time() - start_time
    
    # Aggregate metrics across processes
    avg_loss = running_loss / len(trainloader)
    accuracy = 100. * correct / total
    
    metrics = torch.tensor([avg_loss, accuracy, epoch_time], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= dist.get_world_size()
    
    return {
        'loss': metrics[0].item(),
        'accuracy': metrics[1].item(),
        'time': metrics[2].item()
    }


def validate_distributed(model, testloader, criterion, device, rank):
    """Validate the model (distributed)"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    
    metrics = torch.tensor([avg_loss, accuracy], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= dist.get_world_size()
    
    return {
        'loss': metrics[0].item(),
        'accuracy': metrics[1].item()
    }


def run_distributed_training(args):
    """Run distributed training"""
    rank, world_size = setup_distributed()
    device = get_device_distributed(rank)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("DISTRIBUTED TRAINING")
        print("=" * 70)
        print(f"World Size: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Effective Batch Size: {args.batch_size * world_size}")
        
    # Linear scaling rule
    scaled_lr = args.lr * world_size
    
    if rank == 0:
        print(f"Base LR: {args.lr}, Scaled LR: {scaled_lr}")
        create_directories()
    
    # Data
    trainloader, testloader, train_sampler = get_cifar10_loaders(
        args.batch_size, args.num_workers,
        distributed=True, rank=rank, world_size=world_size
    )
    
    # Model
    model = create_resnet50(device=device)
    model = DDP(model)
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=scaled_lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 8], gamma=0.1
    )
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': [], 'learning_rate': []
    }
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70 + "\n")
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_metrics = train_epoch_distributed(
            model, trainloader, criterion, optimizer, device, rank, epoch, args.epochs
        )
        val_metrics = validate_distributed(model, testloader, criterion, device, rank)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if rank == 0:
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['epoch_time'].append(train_metrics['time'])
            history['learning_rate'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Epoch Time: {train_metrics['time']:.2f}s | LR: {current_lr}")
            print("-" * 70)
    
    total_time = time.time() - total_start
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
        print("=" * 70)
        
        # Save results
        checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': {
                'world_size': world_size,
                'batch_size_per_gpu': args.batch_size,
                'effective_batch_size': args.batch_size * world_size,
                'learning_rate': scaled_lr,
                'total_time': total_time
            }
        }
        
        torch.save(checkpoint, 'checkpoints/distributed_resnet50.pth')
        
        with open('results/distributed_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Metrics
        total_samples = len(trainloader.dataset) * args.epochs
        throughput = total_samples / total_time
        
        metrics = {
            'world_size': world_size,
            'total_time': total_time,
            'avg_epoch_time': total_time / args.epochs,
            'throughput': throughput,
            'final_val_acc': history['val_acc'][-1],
            'batch_size_per_gpu': args.batch_size,
            'effective_batch_size': args.batch_size * world_size
        }
        
        with open('results/distributed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Results saved")
        
        # Generate plots
        generate_training_plots(history, 'distributed')
    
    cleanup_distributed()


# ============================================================================
# SECTION 6: TESTING AND EVALUATION
# ============================================================================

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    model = create_resnet50(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def evaluate_model(model, testloader, device, model_name="Model"):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }


def run_testing(args):
    """Run comprehensive testing and analysis"""
    print("\n" + "=" * 70)
    print("TESTING AND PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    device, device_name = check_environment()
    
    # Load test data
    _, testloader, _ = get_cifar10_loaders(100, 2, distributed=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load baseline results
    baseline_loaded = False
    if os.path.exists('results/baseline_training_history.json'):
        with open('results/baseline_training_history.json', 'r') as f:
            baseline_history = json.load(f)
        with open('results/baseline_metrics.json', 'r') as f:
            baseline_metrics = json.load(f)
        baseline_loaded = True
        print("\n✓ Baseline results loaded")
    
    # Load distributed results
    distributed_loaded = False
    if os.path.exists('results/distributed_history.json'):
        with open('results/distributed_history.json', 'r') as f:
            distributed_history = json.load(f)
        with open('results/distributed_metrics.json', 'r') as f:
            distributed_metrics = json.load(f)
        distributed_loaded = True
        print("✓ Distributed results loaded")
    
    # Evaluate baseline model
    if os.path.exists('checkpoints/baseline_resnet50_cifar10.pth'):
        model, _ = load_model('checkpoints/baseline_resnet50_cifar10.pth', device)
        baseline_eval = evaluate_model(model, testloader, device, "Baseline Model")
        
        # Per-class accuracy
        analyze_per_class_accuracy(baseline_eval, classes)
        
        # Confusion matrix
        plot_confusion_matrix(baseline_eval, classes)
        
        # Classification report
        report = classification_report(
            baseline_eval['targets'], baseline_eval['predictions'],
            target_names=classes, digits=3
        )
        print("\nCLASSIFICATION REPORT")
        print("=" * 70)
        print(report)
        print("=" * 70)
        
        with open('results/classification_report.txt', 'w') as f:
            f.write(report)
    
    # Comparison
    if baseline_loaded and distributed_loaded:
        compare_results(baseline_history, baseline_metrics,
                       distributed_history, distributed_metrics)
    
    print("\n✓ Testing complete! Check results/ directory for outputs.")


def analyze_per_class_accuracy(eval_results, classes):
    """Analyze and plot per-class accuracy"""
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for pred, target in zip(eval_results['predictions'], eval_results['targets']):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1
    
    class_accuracy = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(10)]
    
    # Print table
    print("\nPer-Class Accuracy:")
    print("=" * 60)
    for cls, acc in zip(classes, class_accuracy):
        print(f"  {cls:10s}: {acc:6.2f}%")
    print("=" * 60)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, class_accuracy, color='skyblue', edgecolor='navy', alpha=0.7)
    
    for i, bar in enumerate(bars):
        if class_accuracy[i] >= 80:
            bar.set_color('green')
        elif class_accuracy[i] >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=eval_results['accuracy'], color='r', linestyle='--',
                label=f'Overall: {eval_results["accuracy"]:.2f}%', linewidth=2)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy on CIFAR-10', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: results/per_class_accuracy.png")


def plot_confusion_matrix(eval_results, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(eval_results['targets'], eval_results['predictions'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/confusion_matrix.png")


def compare_results(baseline_history, baseline_metrics, distributed_history, distributed_metrics):
    """Compare baseline vs distributed performance"""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    baseline_time = baseline_metrics['total_training_time_seconds']
    distributed_time = distributed_metrics['total_time']
    speedup = baseline_time / distributed_time
    world_size = distributed_metrics['world_size']
    efficiency = (speedup / world_size) * 100
    
    print(f"\nTraining Time:")
    print(f"  Baseline: {baseline_time:.2f}s ({baseline_time/60:.2f} min)")
    print(f"  Distributed ({world_size} processes): {distributed_time:.2f}s ({distributed_time/60:.2f} min)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.1f}%")
    
    print(f"\nThroughput:")
    print(f"  Baseline: {baseline_metrics['throughput_images_per_second']:.2f} images/s")
    print(f"  Distributed: {distributed_metrics['throughput']:.2f} images/s")
    
    print(f"\nAccuracy:")
    print(f"  Baseline: {baseline_metrics['final_val_accuracy']:.2f}%")
    print(f"  Distributed: {distributed_metrics['final_val_acc']:.2f}%")
    print(f"  Difference: {abs(baseline_metrics['final_val_accuracy'] - distributed_metrics['final_val_acc']):.2f}%")
    print("=" * 70)
    
    # Comparison table
    comparison_df = pd.DataFrame({
        'Metric': ['Configuration', 'Training Time (s)', 'Throughput (img/s)',
                   'Final Val Acc (%)', 'Speedup', 'Efficiency (%)'],
        'Baseline': [
            '1 GPU/MPS',
            f"{baseline_time:.2f}",
            f"{baseline_metrics['throughput_images_per_second']:.2f}",
            f"{baseline_metrics['final_val_accuracy']:.2f}",
            '1.00x',
            '100.0'
        ],
        'Distributed': [
            f"{world_size} processes",
            f"{distributed_time:.2f}",
            f"{distributed_metrics['throughput']:.2f}",
            f"{distributed_metrics['final_val_acc']:.2f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}"
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv('results/performance_comparison.csv', index=False)
    print("\n✓ Saved: results/performance_comparison.csv")
    
    # Plot comparison
    plot_comparison(baseline_history, distributed_history, speedup, efficiency, world_size)


def plot_comparison(baseline_hist, distributed_hist, speedup, efficiency, world_size):
    """Plot training curves comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = range(1, len(baseline_hist['train_loss']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs, baseline_hist['train_loss'], 'b-o', label='Baseline', linewidth=2)
    axes[0, 0].plot(epochs, distributed_hist['train_loss'], 'r-s', label='Distributed', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[0, 1].plot(epochs, baseline_hist['val_acc'], 'b-o', label='Baseline', linewidth=2)
    axes[0, 1].plot(epochs, distributed_hist['val_acc'], 'r-s', label='Distributed', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Epoch Time
    axes[1, 0].plot(epochs, baseline_hist['epoch_time'], 'b-o', label='Baseline', linewidth=2)
    axes[1, 0].plot(epochs, distributed_hist['epoch_time'], 'r-s', label='Distributed', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speedup Analysis
    configs = ['Baseline', f'{world_size} Processes']
    speedups = [1.0, speedup]
    bars = axes[1, 1].bar(configs, speedups, color=['skyblue', 'coral'], edgecolor='black', alpha=0.7)
    axes[1, 1].axhline(y=world_size, color='green', linestyle='--',
                       label=f'Linear ({world_size}x)', linewidth=2)
    axes[1, 1].set_ylabel('Speedup', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Training Speedup', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, speedups):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/baseline_vs_distributed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/baseline_vs_distributed_comparison.png")


# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

def generate_training_plots(history, prefix='baseline'):
    """Generate training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Epoch Time
    axes[1, 0].plot(epochs, history['epoch_time'], 'g-o', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Epoch Time', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'm-o', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{prefix}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: results/{prefix}_training_curves.png")


# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Assignment 2: Distributed ResNet-50 Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline training
  python complete_implementation.py --mode baseline --epochs 10
  
  # Distributed training (2 processes)
  torchrun --nproc_per_node=2 complete_implementation.py --mode distributed --epochs 10
  
  # Testing only
  python complete_implementation.py --mode test
  
  # Run everything
  python complete_implementation.py --mode all
        """
    )
    
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'distributed', 'test', 'all'],
                       help='Execution mode')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (per GPU for distributed)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ASSIGNMENT 2: DISTRIBUTED RESNET-50 TRAINING")
    print("Scalable Data-Parallel Training with Ring-AllReduce")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if args.mode == 'baseline':
        run_baseline_training(args)
    
    elif args.mode == 'distributed':
        run_distributed_training(args)
    
    elif args.mode == 'test':
        run_testing(args)
    
    elif args.mode == 'all':
        print("\nRunning complete pipeline...")
        print("\n>>> Step 1/3: Baseline Training")
        run_baseline_training(args)
        
        print("\n>>> Step 2/3: Testing")
        run_testing(args)
        
        print("\n>>> Step 3/3: Distributed training requires separate launch")
        print("Run: torchrun --nproc_per_node=2 complete_implementation.py --mode distributed")
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE!")
    print("=" * 70)
    print("\nNext steps for assignment:")
    print("  1. Review results in 'results/' directory")
    print("  2. Compile findings into PDF report")
    print("  3. Upload code to GitHub")
    print("  4. Include GitHub link in submission")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
