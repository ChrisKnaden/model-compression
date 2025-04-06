import time
import torch
import torch.nn as nn

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
    return acc, avg_loss

def measure_inference_time(model, dataloader, device, num_batches=100):
    model.eval()

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

            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()

            total_time += (end_time - start_time)

    avg_time_per_batch = total_time / num_batches
    return avg_time_per_batch
