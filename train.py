"""
Main training script for linguistically-aware EWC experiment.

This script runs the complete experiment:
1. Train on Bengali sentiment analysis
2. Train on Hindi sentiment analysis with different CL methods
3. Evaluate forgetting and compare methods
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import random
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

from config import ExperimentConfig, get_linguistic_similarity
from dataset import prepare_dataloaders
from ewc import EWC, LinguisticEWC
from evaluation import evaluate_model, ExperimentTracker


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    ewc_loss_fn: Optional[callable] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        ewc_loss_fn: Optional function to compute EWC loss

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_ewc_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        task_loss = outputs.loss

        # Add EWC loss if applicable
        if ewc_loss_fn is not None:
            ewc_loss = ewc_loss_fn()
            loss = task_loss + ewc_loss
            total_ewc_loss += ewc_loss.item()
        else:
            loss = task_loss
            ewc_loss = torch.tensor(0.0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'ewc_loss': total_ewc_loss / n_batches if ewc_loss_fn else 0.0
    }


def train_language(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    num_epochs: int,
    ewc_loss_fn: Optional[callable] = None
):
    """Train on a single language."""
    for epoch in range(num_epochs):
        metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, ewc_loss_fn
        )
        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"Loss={metrics['total_loss']:.4f}, "
              f"Task Loss={metrics['task_loss']:.4f}, "
              f"EWC Loss={metrics['ewc_loss']:.4f}")


def run_experiment_for_method(
    method: str,
    config: ExperimentConfig,
    dataloaders: Dict[str, Dict[str, DataLoader]],
    languages: List[str],
    device: str,
    tracker: ExperimentTracker
):
    """
    Run the experiment for a single continual learning method.

    Args:
        method: Method name ('naive', 'ewc', 'ling_ewc', 'random_ewc')
        config: Experiment configuration
        dataloaders: DataLoaders for all languages
        languages: List of languages in training order
        device: Device to train on
        tracker: Experiment tracker for logging
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {method.upper()}")
    print(f"{'='*60}")

    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2  # Binary sentiment
    ).to(device)

    # Initialize EWC if needed
    if method in ['ewc', 'ling_ewc', 'random_ewc']:
        if method == 'ling_ewc':
            ewc_module = LinguisticEWC(model, device)
        else:
            ewc_module = EWC(model, device)
    else:
        ewc_module = None

    # Training loop over languages
    step = 0
    for lang_idx, language in enumerate(languages):
        print(f"\n--- Training on {language.upper()} (step {lang_idx}) ---")

        train_loader = dataloaders[language]['train']

        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.num_epochs_per_language
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Define EWC loss function if applicable
        if ewc_module and len(ewc_module.previous_tasks) > 0:
            if method == 'ling_ewc':
                # Linguistic EWC: scale by similarity
                ewc_loss_fn = lambda: ewc_module.compute_linguistic_ewc_loss(
                    current_language=language,
                    ewc_lambda=config.ewc_lambda,
                    similarity_fn=get_linguistic_similarity,
                    invert_similarity=True
                )
            elif method == 'random_ewc':
                # Random scaling (sanity check)
                random_scale = {
                    task['language']: random.uniform(0.0, 1.0)
                    for task in ewc_module.previous_tasks
                }
                print(f"  Random scales: {random_scale}")
                ewc_loss_fn = lambda: ewc_module.compute_ewc_loss(
                    config.ewc_lambda, random_scale
                )
            else:
                # Standard EWC: no scaling
                ewc_loss_fn = lambda: ewc_module.compute_ewc_loss(config.ewc_lambda)
        else:
            ewc_loss_fn = None

        # Train on current language
        train_language(
            model, train_loader, optimizer, scheduler, device,
            config.num_epochs_per_language, ewc_loss_fn
        )

        # Evaluate on all languages seen so far
        print(f"\n  Evaluating on all languages after {language} training:")
        for eval_lang in languages[:lang_idx+1]:
            eval_loader = dataloaders[eval_lang]['eval']
            metrics = evaluate_model(model, eval_loader, device)
            print(f"    {eval_lang}: Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}")

            # Track metrics
            tracker.update(method, eval_lang, step, metrics)

        # Save task for EWC (after evaluation, using eval set for Fisher)
        if ewc_module:
            fisher_loader = dataloaders[language]['train']  # Use eval set for Fisher
            ewc_module.save_task(
                fisher_loader,
                language,
                sample_size=config.fisher_sample_size
            )

        step += 1

    print(f"\nCompleted {method}")


def main():
    """Main experiment runner."""
    # Configuration
    config = ExperimentConfig()
    set_seed(config.random_seed)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Languages in training order
    languages = ['bengali', 'hindi']

    # Load tokenizer
    print(f"\nLoading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare dataloaders
    print("\nPreparing datasets...")
    dataloaders = prepare_dataloaders(
        languages, tokenizer, config, use_demo_data=False
    )

    # Initialize experiment tracker
    tracker = ExperimentTracker(config.methods, languages)

    # Run experiment for each method
    for method in config.methods:
        try:
            run_experiment_for_method(
                method, config, dataloaders, languages, device, tracker
            )
        except Exception as e:
            print(f"\nError in {method}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save and display results
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    tracker.save(config.output_dir)

    # Print key findings
    comparison = tracker.compare_methods()
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    if 'ling_ewc_improvement_over_ewc' in comparison:
        improvement = comparison['ling_ewc_improvement_over_ewc']
        print(f"\nLinguistic EWC vs Standard EWC:")
        print(f"  Reduction in forgetting: {improvement:.4f}")

        if improvement > 0.01:
            print(f"\n  ✓ RESULT: Linguistic similarity scaling REDUCES forgetting")
            print(f"    This suggests that linguistic structure matters for continual learning.")
        elif improvement < -0.01:
            print(f"\n  ✗ RESULT: Linguistic similarity scaling INCREASES forgetting")
            print(f"    This suggests the scaling strategy may need refinement.")
        else:
            print(f"\n  = RESULT: No significant difference between methods")
            print(f"    This suggests Fisher Information alone may be sufficient.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
