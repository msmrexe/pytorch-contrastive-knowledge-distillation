import argparse
import logging
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
import open_clip

from src.config import get_configs
from src.data_loader import get_dataloaders
from src.model import CandidateModel
from src.teacher import get_teacher_model, freeze_model
from src.loss import SymmetricClipLoss, AlternativeClipLoss
from src.engine import train_one_epoch, evaluate
from src.utils import setup_logging, get_device, calc_percentile_tokens

def main(args):
    """
    Main function to run the knowledge distillation training and evaluation.
    """
    # --- 1. Setup ---
    setup_logging()
    logger = logging.getLogger(__name__)
    
    device = get_device(args.device)
    configs = get_configs()
    
    # Update configs with command-line arguments
    configs['device'] = device
    configs['epochs'] = args.epochs
    configs['lr'] = args.lr
    configs['batch_size'] = args.batch_size
    
    logger.info(f"Starting experiment with configs: {configs}")
    logger.info(f"Using device: {device}")

    # --- 2. Load Data ---
    logger.info("Loading datasets...")
    dataset_train, dataset_val = get_dataloaders(configs)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=configs['batch_size'], 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=configs['batch_size'], 
        shuffle=False
    )
    logger.info("Dataloaders created.")

    # --- 3. Load Tokenizers and Models ---
    logger.info("Loading tokenizers...")
    reference_tokenizer = open_clip.get_tokenizer(configs["reference_checkPoint"])
    candidate_tokenizer = AutoTokenizer.from_pretrained(configs["candidate_checkpoint"])
    
    # Calculate token percentiles
    logger.info("Calculating token percentiles for dynamic padding...")
    configs["fa_tok_percentile"] = calc_percentile_tokens(
        dataset_train, candidate_tokenizer, configs["persian"]
    )
    configs["en_tok_percentile"] = calc_percentile_tokens(
        dataset_train, reference_tokenizer, configs["english"]
    )
    logger.info(f"Persian (student) max length: {configs['fa_tok_percentile']}")
    logger.info(f"English (teacher) max length: {configs['en_tok_percentile']}")

    logger.info("Loading models...")
    teacher_model = get_teacher_model(configs).to(device)
    teacher_model = freeze_model(teacher_model)
    teacher_model.eval()
    
    student_model = CandidateModel(
        model_name=configs["candidate_checkpoint"],
        embedding_dim=configs["candidate_embedding"],
        projection_dim=configs["project_to"],
        dropout=configs["dropout"]
    ).to(device)
    
    logger.info("Models loaded and teacher model frozen.")

    # --- 4. Setup Loss, Optimizer, and Scheduler ---
    
    # Learnable temperature parameter
    temperature = torch.nn.Parameter(torch.tensor(configs['temperature']).float()).to(device)
    
    # Select loss function
    if args.loss_type == 'symmetric':
        loss_fn = SymmetricClipLoss(temperature)
        logger.info("Using Symmetric CLIP Loss.")
    elif args.loss_type == 'alternative':
        loss_fn = AlternativeClipLoss(temperature)
        logger.info("Using Alternative (Simplified) CLIP Loss.")
    else:
        logger.error(f"Invalid loss type: {args.loss_type}")
        raise ValueError("Invalid loss type. Choose 'symmetric' or 'alternative'.")

    # Optimizer (includes learnable temperature)
    optimizer = optim.AdamW(
        list(student_model.parameters()) + [temperature], 
        weight_decay=configs["weight_decay"], 
        lr=configs['lr']
    )
    
    # Scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 'max', 
        patience=configs['patience'], 
        factor=configs['factor']
    )
    
    # --- 5. Training Loop ---
    logger.info("Starting training...")
    best_val_acc = float('-inf')
    metrics = pd.DataFrame(columns=["Avg-train-loss", "Avg-train-accuracy", "Avg-val-loss", "Avg-val-accuracy"])

    for epoch in range(configs['epochs']):
        logger.info(f"--- Epoch {epoch+1}/{configs['epochs']} ---")
        
        train_loss, train_acc = train_one_epoch(
            student_model,
            teacher_model,
            train_loader,
            loss_fn,
            optimizer,
            candidate_tokenizer,
            reference_tokenizer,
            configs
        )
        
        val_loss, val_acc = evaluate(
            student_model,
            teacher_model,
            val_loader,
            loss_fn,
            candidate_tokenizer,
            reference_tokenizer,
            configs
        )
        
        metrics.loc[epoch+1] = [train_loss, train_acc, val_loss, val_acc]
        
        logger.info(f"Epoch {epoch+1} Metrics:")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%")
        logger.info(f"Current Temperature: {temperature.item():.4f}")
        
        lr_scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), configs['save_path'])
            logger.info(f"New best model saved with accuracy: {best_val_acc*100:.2f}%")

    logger.info("Training complete.")
    logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Metrics summary:\n{metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Knowledge Distillation Training")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--loss_type', type=str, default='symmetric', 
                        choices=['symmetric', 'alternative'], 
                        help='Type of CLIP loss to use ("symmetric" or "alternative")')
    
    args = parser.parse_args()
    main(args)
