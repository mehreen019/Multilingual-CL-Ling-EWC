"""
Evaluation utilities for continual learning experiments.

This module provides:
1. Model evaluation on test sets
2. Forgetting metric computation
3. Results tracking and visualization
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm
import json
import os


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary with metrics (accuracy, loss)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    model.train()
    return {
        'accuracy': accuracy,
        'loss': avg_loss
    }


class ContinualLearningMetrics:
    """
    Track and compute continual learning metrics.

    Key metrics:
    1. Performance on each language after each training step
    2. Forgetting: drop in performance on previous languages
    3. Transfer: performance gain on new languages
    """

    def __init__(self, languages: List[str]):
        """
        Initialize metrics tracker.

        Args:
            languages: List of language codes in training order
        """
        self.languages = languages
        self.history = {lang: [] for lang in languages}  # Performance history per language
        self.peak_performance = {}  # Best performance seen for each language

    def update(self, language: str, step: int, metrics: Dict[str, float]):
        """
        Update metrics after a training step.

        Args:
            language: Language being evaluated
            step: Training step number
            metrics: Evaluation metrics
        """
        entry = {
            'step': step,
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss']
        }
        self.history[language].append(entry)

        # Update peak performance
        if language not in self.peak_performance:
            self.peak_performance[language] = metrics['accuracy']
        else:
            self.peak_performance[language] = max(
                self.peak_performance[language],
                metrics['accuracy']
            )

    def compute_forgetting(self) -> Dict[str, float]:
        """
        Compute forgetting metric for each language.

        Forgetting is defined as:
        f_i = max_j(acc_i,j) - acc_i,final

        where acc_i,j is accuracy on language i after training step j.

        Returns:
            Dictionary mapping language to forgetting value
        """
        forgetting = {}

        for language in self.languages:
            if not self.history[language]:
                continue

            accuracies = [entry['accuracy'] for entry in self.history[language]]
            if len(accuracies) < 2:
                forgetting[language] = 0.0
            else:
                peak_acc = max(accuracies)
                final_acc = accuracies[-1]
                forgetting[language] = peak_acc - final_acc

        return forgetting

    def compute_average_forgetting(self) -> float:
        """
        Compute average forgetting across all languages except the last one.

        This is the primary metric for continual learning evaluation.

        Returns:
            Average forgetting value
        """
        forgetting_dict = self.compute_forgetting()

        # Exclude the last language (it wasn't learned before, so no forgetting)
        languages_to_consider = self.languages[:-1]
        if not languages_to_consider:
            return 0.0

        forgetting_values = [
            forgetting_dict.get(lang, 0.0)
            for lang in languages_to_consider
        ]

        return sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0

    def get_final_performance(self) -> Dict[str, float]:
        """
        Get final performance on all languages.

        Returns:
            Dictionary mapping language to final accuracy
        """
        final_perf = {}
        for language in self.languages:
            if self.history[language]:
                final_perf[language] = self.history[language][-1]['accuracy']
            else:
                final_perf[language] = 0.0

        return final_perf

    def compute_backward_transfer(self, language: str) -> float:
        """
        Compute backward transfer for a language.

        Backward transfer measures how learning new languages affects
        performance on this language:
        BWT_i = acc_i,final - acc_i,i

        Positive BWT = beneficial transfer, Negative BWT = forgetting

        Args:
            language: Language to compute BWT for

        Returns:
            Backward transfer value
        """
        lang_idx = self.languages.index(language)
        history = self.history[language]

        if len(history) < 2:
            return 0.0

        # Accuracy right after training on this language
        acc_after_training = history[lang_idx]['accuracy']

        # Final accuracy
        acc_final = history[-1]['accuracy']

        return acc_final - acc_after_training

    def summary(self) -> Dict:
        """
        Generate summary of all metrics.

        Returns:
            Dictionary with comprehensive metrics
        """
        forgetting_dict = self.compute_forgetting()
        avg_forgetting = self.compute_average_forgetting()
        final_perf = self.get_final_performance()
        avg_final_acc = sum(final_perf.values()) / len(final_perf) if final_perf else 0.0

        return {
            'forgetting_per_language': forgetting_dict,
            'average_forgetting': avg_forgetting,
            'final_performance': final_perf,
            'average_final_accuracy': avg_final_acc,
            'peak_performance': self.peak_performance
        }

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            'languages': self.languages,
            'history': self.history,
            'peak_performance': self.peak_performance,
            'summary': self.summary()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to {filepath}")


class ExperimentTracker:
    """
    Track results across multiple experimental conditions.
    """

    def __init__(self, methods: List[str], languages: List[str]):
        """
        Initialize experiment tracker.

        Args:
            methods: List of method names ('naive', 'ewc', etc.)
            languages: List of languages in training order
        """
        self.methods = methods
        self.languages = languages
        self.metrics = {
            method: ContinualLearningMetrics(languages)
            for method in methods
        }

    def update(self, method: str, language: str, step: int, metrics: Dict[str, float]):
        """Update metrics for a specific method."""
        self.metrics[method].update(language, step, metrics)

    def compare_methods(self) -> Dict:
        """
        Compare all methods.

        Returns:
            Dictionary with comparative analysis
        """
        comparison = {}

        for method in self.methods:
            comparison[method] = self.metrics[method].summary()

        # Compute relative improvements
        if 'naive' in comparison and 'ewc' in comparison:
            naive_forget = comparison['naive']['average_forgetting']
            ewc_forget = comparison['ewc']['average_forgetting']
            comparison['ewc_improvement_over_naive'] = naive_forget - ewc_forget

        if 'ewc' in comparison and 'ling_ewc' in comparison:
            ewc_forget = comparison['ewc']['average_forgetting']
            ling_ewc_forget = comparison['ling_ewc']['average_forgetting']
            comparison['ling_ewc_improvement_over_ewc'] = ewc_forget - ling_ewc_forget

        return comparison

    def save(self, output_dir: str):
        """Save all results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save individual method metrics
        for method in self.methods:
            filepath = os.path.join(output_dir, f'{method}_metrics.json')
            self.metrics[method].save(filepath)

        # Save comparison
        comparison = self.compare_methods()
        comparison_path = os.path.join(output_dir, 'comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nExperiment results saved to {output_dir}")
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)

        for method in self.methods:
            summary = comparison[method]
            print(f"\n{method.upper()}:")
            print(f"  Average Forgetting: {summary['average_forgetting']:.4f}")
            print(f"  Final Avg Accuracy: {summary['average_final_accuracy']:.4f}")
            print(f"  Final Performance: {summary['final_performance']}")

        if 'ling_ewc_improvement_over_ewc' in comparison:
            improvement = comparison['ling_ewc_improvement_over_ewc']
            print(f"\nLinguistic EWC improvement over standard EWC: {improvement:.4f}")
            if improvement > 0:
                print("✓ Linguistic scaling REDUCES forgetting")
            else:
                print("✗ Linguistic scaling does NOT reduce forgetting")
