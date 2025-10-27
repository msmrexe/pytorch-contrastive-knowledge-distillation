import logging
import sys
import torch
import numpy as np
import os

def setup_logging():
    """
    Configures logging to output to both console and a file.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/distillation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_device(which="cuda:0"):
    """
    Selects the appropriate device (GPU or CPU) for computation.
    """
    if torch.cuda.is_available():
        device = torch.device(which)
    else:
        logging.warning("CUDA not available, falling back to CPU. This will be very slow.")
        device = torch.device("cpu")
    return device

def get_cls_token(tensor, cls_token_index=0):
    """
    Extracts the classification (CLS) token from the input tensor.
    Shape: (batch_size, seq_length, hidden_dim) -> (batch_size, 1, hidden_dim)
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D tensor")
    return tensor[:, cls_token_index, :].unsqueeze(1)

def flatten_middle(tensor):
    """
    Flattens the middle dimension (sequence length) of the input tensor.
    Shape: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
    """
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError(f"Expected 3D tensor with middle dim 1, got {tensor.shape}")
    return tensor.view(tensor.size(0), -1)

def calc_percentile_tokens(dataset, tokenizer, field, percentile=99, threshold=1):
    """
    Calculate the token length at a specific percentile for a dataset field.
    """
    try:
        tokenized = tokenizer(dataset[field])
        if not isinstance(tokenized, torch.Tensor):
            token_lengths = [len(sen) for sen in tokenized['input_ids']]
        else:
            token_lengths = [tensor.nonzero().size(0) for tensor in tokenized]
        
        percentile_length = np.percentile(token_lengths, percentile)
        return int(percentile_length) + threshold
    except Exception as e:
        logging.error(f"Error calculating percentile tokens for field '{field}': {e}")
        # Fallback to a safe default
        return 128
