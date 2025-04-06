import torch
from tqdm import tqdm
from contextlib import nullcontext

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Loss wrapper for KD
class KDParams:
    def __init__(self, alpha=0.9, temperature=4.0):
        self.alpha = alpha
        self.temperature = temperature

def loss_fn_kd(student_outputs, labels, teacher_outputs, params):
    import torch.nn.functional as F

    T = params.temperature
    alpha = params.alpha

    kd_loss = F.kl_div(
        F.log_softmax(student_outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1),
        reduction='batchmean'
    ) * (alpha * T * T)

    ce_loss = F.cross_entropy(student_outputs, labels) * (1. - alpha)
    return kd_loss + ce_loss

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10, use_amp=False):
    model.to(device)

    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda'))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")


def train_model_kd(student_model, teacher_model, train_loader, optimizer, device,
                    kd_params, loss_fn_kd, num_epochs=10, use_amp=False):

    # Disable AMP for MPS/CPU
    if device.type != 'cuda':
        use_amp = False

    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda'))

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Choose context: autocast for CUDA, nullcontext otherwise
            amp_context = autocast(device_type='cuda') if use_amp and device.type == 'cuda' else nullcontext()

            with amp_context:
                student_outputs = student_model(inputs)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                loss = loss_fn_kd(student_outputs, targets, teacher_outputs, kd_params)

            if use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")