import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricClipLoss(nn.Module):
    """
    Implements the symmetric CLIP loss.
    (Loss_Ref_to_Cand + Loss_Cand_to_Ref) / 2
    """
    def __init__(self, temperature_param):
        super().__init__()
        self.temperature = temperature_param

    def forward(self, reference_embeds, candidate_embeds):
        # Normalize embeddings
        reference_embeds_norm = F.normalize(reference_embeds, p=2, dim=-1)
        candidate_embeds_norm = F.normalize(candidate_embeds, p=2, dim=-1)
        
        # Calculate logits (scaled cosine similarity)
        # Shape: (batch_size, batch_size)
        logits = torch.matmul(reference_embeds_norm, candidate_embeds_norm.T) / self.temperature
        
        # Create targets (identity matrix)
        batch_size = reference_embeds.size(0)
        device = reference_embeds.device
        targets = torch.arange(batch_size, device=device)

        # Compute symmetric cross-entropy loss
        loss_ref_to_cand = F.cross_entropy(logits, targets)
        loss_cand_to_ref = F.cross_entropy(logits.T, targets)
        loss = (loss_ref_to_cand + loss_cand_to_ref) / 2
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        corrects = (predictions == targets).sum().item()
        
        return loss, corrects

class AlternativeClipLoss(nn.Module):
    """
    Implements the alternative (simplified) CLIP loss.
    Loss_Ref_to_Cand only.
    """
    def __init__(self, temperature_param):
        super().__init__()
        self.temperature = temperature_param

    def forward(self, reference_embeds, candidate_embeds):
        # Normalize embeddings
        reference_embeds_norm = F.normalize(reference_embeds, p=2, dim=-1)
        candidate_embeds_norm = F.normalize(candidate_embeds, p=2, dim=-1)
        
        # Calculate logits (scaled cosine similarity)
        logits = torch.matmul(reference_embeds_norm, candidate_embeds_norm.T) / self.temperature
        
        # Create targets
        batch_size = reference_embeds.size(0)
        device = reference_embeds.device
        targets = torch.arange(batch_size, device=device)

        # Compute single cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        corrects = (predictions == targets).sum().item()
        
        return loss, corrects
