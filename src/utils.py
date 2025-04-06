def count_total_parameters(model, verbose=True):
    """Counts and optionally prints the total number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total number of parameters in the model: {total}")
    return total

import torch

def load_model(filepath, device=None, weights_only=False):
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(filepath, map_location=device)

    if weights_only:
        return checkpoint.get("model_state_dict", checkpoint)
    return checkpoint
