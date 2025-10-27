import torch.nn as nn
from open_clip import model as TE_Model
import logging

def get_teacher_model(configs):
    """
    Initializes and returns the pre-trained TextTransformer (teacher model).
    """
    logger = logging.getLogger(__name__)
    try:
        model = TE_Model.TextTransformer(
            context_length=configs['reference_context_length'],
            vocab_size=configs["reference_vocab_size"],
            width=configs["reference_width"],
            layers=configs["reference_layers"],
            heads=configs["reference_heads"],
            output_dim=configs["reference_embedding"]
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize teacher model: {e}")
        raise

def freeze_model(model):
    """
    Freezes all parameters of a given model.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Freezing model: {model.__class__.__name__}")
    for param in model.parameters():
        param.requires_grad = False
    return model
