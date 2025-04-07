from pathlib import Path
from src.evaluate import evaluate
import torch
import time
import torch_pruning as tp



def count_total_parameters(model, verbose=True):
    """Counts and optionally prints the total number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total number of parameters in the model: {total}")
    return total

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

def iterative_pruner(pruner, iterative_pruning_steps=1):
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
        if isinstance(imp, tp.importance.TaylorImportance):
            # TaylorImportance needs gradient
            loss = pruner.model(example_inputs).sum()
            loss.backward()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)

        # Optional debug output:
        # print(f"After pruning step {i + 1}:")
        # print(f"Parameters: {nparams}, Δparams: {base_nparams - nparams}")

