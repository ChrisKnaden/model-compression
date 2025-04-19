from pathlib import Path
from src.evaluate import evaluate
import torch
import time
import torch_pruning as tp
from torch.ao.nn.quantized import BatchNorm2d

def save_model(model, relative_path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / relative_path
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    torch.save(model, filepath)

def load_model(relative_path, device=None, weights_only=False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / relative_path

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    if weights_only:
        return checkpoint.get("model_state_dict", checkpoint)
    return checkpoint

def save_quantized_model(model, path: str):
    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / path
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, BatchNorm2d):
            if hasattr(module, 'scale') and isinstance(module.scale, torch.Tensor):
                module.scale = torch.tensor(module.scale.item())  # scalar tensor
            if hasattr(module, 'zero_point') and isinstance(module.zero_point, torch.Tensor):
                module.zero_point = torch.tensor(int(module.zero_point.item()))  # scalar tensor

    scripted = torch.jit.script(model)
    scripted.save(filepath)


def load_quantized_model(path: str):
    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / path

    model = torch.jit.load(filepath)
    model.eval()
    return model

def quantize_model(model, val_loader, device, backend='fbgemm'):
    model.to(device)

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Prepare model for quantization
    model_fp32 = model
    model_fp32.eval()

    # Fuse modules (make sure model has a `fuse_model()` method)
    model_fp32.fuse_model()

    # Set quantization config
    model_fp32.qconfig = torch.quantization.get_default_qconfig(backend)

    # Insert observers
    model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    # Calibration with representative dataset
    evaluate(model_prepared, val_loader, device)

    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)

    return model_quantized

def iterative_pruner(pruner, gradient=False, iterative_pruning_steps=1):
    # Set example inputs (same every time)
    example_inputs = torch.randn(1, 3, 32, 32)

    # Set ignored layers (always skip final classifier)
    ignored_layers = []
    for m in pruner.model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
    pruner.ignored_layers = ignored_layers

    # Base metrics
    base_macs, base_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)

    # Get importance criterion
    imp = pruner.importance

    for i in range(iterative_pruning_steps):
        if gradient:
            # for e.g. TaylorImportance needs gradient
            loss = pruner.model(example_inputs).sum()
            loss.backward()

        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)

        # Optional debug output:
        # print(f"After pruning step {i + 1}:")
        # print(f"Parameters: {nparams}, Δparams: {base_nparams - nparams}")

