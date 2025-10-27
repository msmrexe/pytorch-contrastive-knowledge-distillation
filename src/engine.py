import torch
import logging
from tqdm import tqdm

def get_batch_texts(pairs, configs):
    """Helper to extract texts for tokenization."""
    return pairs[configs['persian']], pairs[configs['english']]

def tokenize_batch(pairs, candidate_tokenizer, reference_tokenizer, configs):
    """Tokenizes a batch of paired texts."""
    persian_texts, english_texts = get_batch_texts(pairs, configs)
    device = configs['device']
    
    candidate_tokenized = candidate_tokenizer(
        persian_texts,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        max_length=configs["fa_tok_percentile"]
    ).to(device)
    
    reference_tokenized = reference_tokenizer(
        english_texts
    ).to(device)
    
    return candidate_tokenized, reference_tokenized

def train_one_epoch(student_model, teacher_model, dataloader, loss_fn, optimizer, 
                    candidate_tokenizer, reference_tokenizer, configs):
    """
    Runs a single training epoch.
    """
    student_model.train()
    logger = logging.getLogger(__name__)
    
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for _, pairs in progress_bar:
        optimizer.zero_grad()
        
        try:
            candidate_tokenized, reference_tokenized = tokenize_batch(
                pairs, candidate_tokenizer, reference_tokenizer, configs
            )
            
            # Generate embeddings
            with torch.no_grad(): # Teacher is frozen
                reference_embeds = teacher_model(reference_tokenized)
            
            candidate_embeds = student_model(
                input_ids=candidate_tokenized["input_ids"],
                attention_mask=candidate_tokenized["attention_mask"]
            )
            
            # Compute loss and accuracy
            loss, corrects = loss_fn(reference_embeds, candidate_embeds)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            batch_size = reference_tokenized.size(0)
            total_loss += loss.item() * batch_size
            total_corrects += corrects
            total_samples += batch_size
            
            progress_bar.set_postfix(
                loss=total_loss/total_samples, 
                acc=total_corrects/total_samples
            )

        except Exception as e:
            logger.error(f"Error during training step: {e}")
            continue # Skip batch on error

    avg_loss = total_loss / total_samples
    avg_accuracy = total_corrects / total_samples
    
    return avg_loss, avg_accuracy

def evaluate(student_model, teacher_model, dataloader, loss_fn, 
             candidate_tokenizer, reference_tokenizer, configs):
    """
    Runs evaluation on the validation set.
    """
    student_model.eval()
    logger = logging.getLogger(__name__)
    
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    
    with torch.no_grad():
        for _, pairs in progress_bar:
            try:
                candidate_tokenized, reference_tokenized = tokenize_batch(
                    pairs, candidate_tokenizer, reference_tokenizer, configs
                )
                
                # Generate embeddings
                reference_embeds = teacher_model(reference_tokenized)
                candidate_embeds = student_model(
                    input_ids=candidate_tokenized["input_ids"],
                    attention_mask=candidate_tokenized["attention_mask"]
                )
                
                # Compute loss and accuracy
                loss, corrects = loss_fn(reference_embeds, candidate_embeds)
                
                batch_size = reference_tokenized.size(0)
                total_loss += loss.item() * batch_size
                total_corrects += corrects
                total_samples += batch_size
                
                progress_bar.set_postfix(
                    loss=total_loss/total_samples, 
                    acc=total_corrects/total_samples
                )

            except Exception as e:
                logger.error(f"Error during evaluation step: {e}")
                continue # Skip batch on error

    avg_loss = total_loss / total_samples
    avg_accuracy = total_corrects / total_samples
    
    return avg_loss, avg_accuracy
