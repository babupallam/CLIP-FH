import torch


def get_device(verbose=True):
    """
    Returns the best available device ('cuda' if GPU is available, else 'cpu').
    Also prints the device name if verbose is True.

    Usage:
        device = get_device()
        model.to(device)
        tensor = tensor.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] Using CPU (no GPU available)")
    return device
