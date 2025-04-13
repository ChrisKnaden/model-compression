from pathlib import Path
import time
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, val_loader, device, use_half=False):

    criterion = nn.CrossEntropyLoss().to(device)
    model.eval().to(device)

    correct = 0
    total = 0
    total_loss = 0
    start = time.time()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_half:
                inputs = inputs.half()
                criterion = criterion.half()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item() * targets.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / total
    elapsed = time.time() - start

    print(f"Validation Accuracy: {acc:.2f}%, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
    return acc, avg_loss, elapsed

import time
import torch

def measure_inference_time(model, dataloader, device, num_batches=100):
    model.eval()

    # Warm-up
    with torch.inference_mode():
        for _ in range(5):
            inputs, _ = next(iter(dataloader))
            inputs = inputs.to(device)
            _ = model(inputs)

    total_time = 0.0

    with torch.inference_mode():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()

            total_time += (end_time - start_time)

    avg_time_per_batch = total_time / num_batches
    return avg_time_per_batch

def count_total_parameters(model, verbose=True):
    """Counts and optionally prints the total number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total number of parameters in the model: {total}")
    return total

def evaluate_model_all_metrics(model, val_loader, device, path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / path
    filepath.parent.mkdir(parents=True, exist_ok=True)

    accuracy, loss, _ = evaluate(model, val_loader, device)
    params = count_total_parameters(model)
    memory_footprint = estimate_model_memory_footprint_from_bits(params)
    file_size = get_model_file_size(filepath)

    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)

    # This is horrible, only for presentation purpose
    is_quantized = 'quant' in str(path).lower()

    if not is_quantized:
        model_cpu = model_cpu.float()

    inference_time = measure_inference_time(model_cpu, val_loader, cpu_device)
    return {
        "accuracy": accuracy,
        "parameters": params,
        "inference_time": inference_time,
        "memory_footprint": memory_footprint,
        "file_size": file_size
    }

def estimate_model_memory_footprint_from_bits(param_count, bits=32):
    bytes_per_param = bits / 8
    total_bytes = param_count * bytes_per_param
    return total_bytes / (1024 ** 2)

def get_model_file_size(path, unit="MB"):
    size_bytes = os.path.getsize(path)
    if unit == "MB":
        return size_bytes / (1024 * 1024)
    elif unit == "KB":
        return size_bytes / 1024
    else:
        return size_bytes

# CIFAR-10 classes
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_model_predictions(model, dataloader, device='cuda', num_images=8):
    model.to(device)
    model.eval()

    # Get a batch
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Unnormalize for display
    def imshow(img_tensor):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img_tensor.cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')

    # Plot predictions
    plt.figure(figsize=(num_images * 1.5, 3))
    for idx in range(min(num_images, images.size(0))):
        plt.subplot(1, num_images, idx + 1)
        imshow(images[idx])
        pred_label = CLASSES[preds[idx]]
        true_label = CLASSES[labels[idx]]
        title_color = 'green' if preds[idx] == labels[idx] else 'red'
        plt.title(f"P: {pred_label}\nT: {true_label}", color=title_color, fontsize=8)
    plt.tight_layout()
    plt.show()
