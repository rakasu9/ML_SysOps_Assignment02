#!/usr/bin/env python3
"""
Distributed ResNet-50 Training Script
Assignment 2: Distributed Deep Learning with Ring-AllReduce

This script implements data-parallel distributed training using PyTorch DDP.
Run with: torchrun --nproc_per_node=2 distributed_train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import os
import time
import json
import argparse
from datetime import datetime


def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend='gloo')  # Use 'nccl' for NVIDIA GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def get_device(rank):
    """Get device for this process"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def get_dataloaders(rank, world_size, batch_size, num_workers=2):
    """Create distributed data loaders"""
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
    
    train_sampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    
    test_sampler = DistributedSampler(
        testset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader, train_sampler


def create_model(device):
    """Create ResNet-50 model wrapped with DDP"""
    model = torchvision.models.resnet50(weights=None)
    
    # Modify for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    model = model.to(device)
    model = DDP(model)
    
    return model


def train_epoch(model, trainloader, criterion, optimizer, device, rank):
    """Train for one epoch"""
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
    
    # Aggregate metrics
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


def validate(model, testloader, criterion, device, rank):
    """Validate the model"""
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


def main():
    parser = argparse.ArgumentParser(description='Distributed ResNet-50 Training')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='data loading workers')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    device = get_device(rank)
    
    # Apply linear scaling rule
    scaled_lr = args.lr * world_size
    
    if rank == 0:
        print("=" * 70)
        print("DISTRIBUTED TRAINING STARTED")
        print("=" * 70)
        print(f"World Size: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Effective Batch Size: {args.batch_size * world_size}")
        print(f"Base LR: {args.lr}, Scaled LR: {scaled_lr}")
        print(f"Device: {device}")
        print("=" * 70)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data loaders
    trainloader, testloader, train_sampler = get_dataloaders(
        rank, world_size, args.batch_size, args.num_workers
    )
    
    # Model
    model = create_model(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=scaled_lr, 
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 8], gamma=0.1
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': [], 'learning_rate': []
    }
    
    total_start = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(model, trainloader, criterion, optimizer, device, rank)
        
        # Validate
        val_metrics = validate(model, testloader, criterion, device, rank)
        
        # Update LR
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if rank == 0:
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
    
    total_time = time.time() - total_start
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Avg Time/Epoch: {total_time/args.epochs:.2f}s")
        print(f"Final Val Acc: {history['val_acc'][-1]:.2f}%")
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
        
        # Save metrics
        metrics = {
            'world_size': world_size,
            'total_time': total_time,
            'avg_epoch_time': total_time / args.epochs,
            'throughput': (len(trainloader.dataset) * args.epochs) / total_time,
            'final_val_acc': history['val_acc'][-1]
        }
        with open('results/distributed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\n✓ Results saved!")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
