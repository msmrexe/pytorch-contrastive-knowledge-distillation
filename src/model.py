import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from src.utils import get_cls_token, flatten_middle
import logging

class Swish(nn.Module):
    """
    Implements the Swish activation function: x * sigmoid(beta * x)
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class LinearProjection(nn.Module):
    """
    A projection head with Swish activation, batch norm, dropout, 
    and residual connections.
    """
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(LinearProjection, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.swish = Swish(beta=1.0)
        # Use BatchNorm1d for (N, C) or (N, C, L) inputs
        # Here, our input will be (N, C), so C=projection_dim
        self.batch_norm = nn.BatchNorm1d(projection_dim)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        # Input x shape: (batch_size, embedding_dim)
        projected = self.projection(x)
        residual = projected
        
        x = self.swish(projected)
        
        # BatchNorm1d expects (N, C)
        if x.dim() == 2:
            x = self.batch_norm(x)
        else:
            # Handle potential (N, L, C) -> (N, C, L)
            x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.fc(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class CandidateModel(nn.Module):
    """
    The student model (e.g., smaller-LaBSE) with a projection head.
    """
    def __init__(self, model_name, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        try:
            self.configs = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load student model '{model_name}': {e}")
            raise

        self.candidate_projection = LinearProjection(embedding_dim, projection_dim, dropout)
        
        # BatchNorm1d for (N, 1, C) input, applying batchnorm across C
        self.batch_norm_cls = nn.BatchNorm1d(embedding_dim)
        self.targetTokenIdx = 0 # CLS token index

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get CLS token: (batch_size, seq_len, hidden_dim) -> (batch_size, 1, hidden_dim)
        cls_embed = get_cls_token(output.last_hidden_state, self.targetTokenIdx)
        
        # Apply BatchNorm to CLS token embedding
        # Input to BatchNorm1d: (N, C, L) -> (batch_size, hidden_dim, 1)
        cls_embed_bn = self.batch_norm_cls(cls_embed.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Flatten for projection head: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        cls_embed_flat = flatten_middle(cls_embed_bn)
        
        # Project to teacher's dimension
        projected_embed = self.candidate_projection(cls_embed_flat)
        
        return projected_embed
