"""Minimal debug script to check EWC implementation."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW

from config import ExperimentConfig, get_linguistic_similarity
from dataset import prepare_dataloaders
from ewc import LinguisticEWC

def debug_ewc():
    """Debug EWC computation step by step."""
    print("="*60)
    print("EWC DEBUG DIAGNOSTICS (Memory Efficient)")
    print("="*60)
    
    config = ExperimentConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reduce sizes further for debugging
    config.train_size = 200
    config.eval_size = 100
    config.fisher_sample_size = 100
    config.batch_size = 4
    
    print(f"Using device: {device}")
    print(f"Config: train_size={config.train_size}, batch_size={config.batch_size}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    ).to(device)
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    languages = ['bengali', 'hindi']
    dataloaders = prepare_dataloaders(
        languages, tokenizer, config, use_demo_data=False
    )
    
    # Initialize EWC
    ewc = LinguisticEWC(model, device)
    
    print(f"\n1. Training on Bengali (1 epoch only)...")
    train_loader = dataloaders['bengali']['train']
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:  # Only 10 batches
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clear memory
        del input_ids, attention_mask, labels, outputs, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"   Training complete")
    
    # Save Bengali task
    print(f"\n2. Computing Fisher Information...")
    ewc.save_task(dataloaders['bengali']['train'], 'bengali', sample_size=50)
    
    # Check Fisher values
    fisher = ewc.previous_tasks[0]['fisher']
    print(f"\n3. Fisher Information Stats:")
    
    # Check classifier layers only (most important)
    classifier_fisher = {k: v for k, v in fisher.items() if 'classifier' in k}
    for name, values in classifier_fisher.items():
        print(f"   {name}:")
        print(f"     Mean: {values.mean().item():.2e}")
        print(f"     Max: {values.max().item():.2e}")
    
    # Train on Hindi with EWC
    print(f"\n4. Training on Hindi with EWC...")
    hindi_loader = dataloaders['hindi']['train']
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    ewc_losses = []
    task_losses = []
    
    for batch_idx, batch in enumerate(hindi_loader):
        if batch_idx >= 5:  # Only 5 batches
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, labels=labels)
        task_loss = outputs.loss
        
        # Compute EWC loss
        ewc_loss = ewc.compute_linguistic_ewc_loss(
            current_language='hindi',
            ewc_lambda=5000.0,
            similarity_fn=get_linguistic_similarity,
            invert_similarity=True
        )
        
        loss = task_loss + ewc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ewc_losses.append(ewc_loss.item())
        task_losses.append(task_loss.item())
        
        # Clear memory
        del input_ids, attention_mask, labels, outputs, task_loss, ewc_loss, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n5. Results:")
    print(f"   Average Task Loss: {sum(task_losses)/len(task_losses):.4f}")
    print(f"   Average EWC Loss: {sum(ewc_losses)/len(ewc_losses):.4f}")
    
    print("\n" + "="*60)
    if sum(ewc_losses)/len(ewc_losses) > 0.1:
        print("✓ EWC is working! Loss values look reasonable.")
    else:
        print("⚠️  EWC loss is still too small.")
    print("="*60)

if __name__ == "__main__":
    debug_ewc()