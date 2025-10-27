import torch
from src.utils import get_device

def get_configs():
    """
    Returns a dictionary of baseline configurations.
    """
    return {
        "device": get_device(),
        "reference_checkPoint": "EVA02-E-14-plus",           # Teacher
        "candidate_checkpoint": "setu4993/smaller-LaBSE",     # Student
        "train_path": "data/train.csv",
        "val_path": "data/val.csv",
        "save_path": "./best_student_model.pth",
        "english": "en",
        "persian": "fa",
        "batch_size": 128,
        "lr": 1e-4,
        "epochs": 5,
        "tok_percentile": 99,
        "temperature": 20.0,
        "dropout": 0.05,
        "weight_decay": 1e-5,
        "patience": 1,
        "factor": 0.8,
        
        # Model dimensions (can be auto-filled)
        "reference_embedding": 1024,
        "candidate_embedding": 768, # Will be set by AutoConfig in main.py
        "project_to": 1024,
        
        # Teacher model specifics (for open_clip TextEncoder)
        "reference_context_length": 77,
        "reference_vocab_size": 49408,
        "reference_heads": 20,
        "reference_width": 1280,
        "reference_layers": 32,
        
        "cls_token_index": 0,
    }
