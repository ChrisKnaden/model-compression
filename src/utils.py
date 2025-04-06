from pathlib import Path
import torch

def count_total_parameters(model, verbose=True):
    """Counts and optionally prints the total number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total number of parameters in the model: {total}")
    return total

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
