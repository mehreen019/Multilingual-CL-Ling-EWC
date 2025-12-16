"""
Configuration for the linguistically-aware EWC experiment.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for the PoC experiment."""

    # Model configuration
    model_name: str = "xlm-roberta-base"
    max_length: int = 128

    # Training configuration
    num_epochs_per_language: int = 3
    batch_size: int = 8  # REDUCED from 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # EWC configuration
    ewc_lambda: float = 5000.0
    fisher_sample_size: int = 200  # REDUCED from 1000

    # Linguistic similarity configuration
    bengali_hindi_similarity: float = 0.6

    # Experimental conditions
    methods: list = None

    # Data configuration
    train_size: int = 500  # REDUCED from 1000
    eval_size: int = 200
    random_seed: int = 42

    # Output configuration
    output_dir: str = "./results"
    save_models: bool = False  # CHANGED to False to save memory

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['naive', 'ewc', 'ling_ewc', 'random_ewc']


# Linguistic feature-based similarity computation
LANGUAGE_SIMILARITY_MATRIX = {
    ('bengali', 'hindi'): 0.6,
    ('hindi', 'bengali'): 0.6,
    ('bengali', 'bengali'): 1.0,
    ('hindi', 'hindi'): 1.0,
}


def get_linguistic_similarity(lang1: str, lang2: str) -> float:
    """
    Get linguistic similarity score between two languages.

    Args:
        lang1: First language code
        lang2: Second language code

    Returns:
        Similarity score in [0, 1], where 1 is most similar
    """
    key = (lang1.lower(), lang2.lower())
    return LANGUAGE_SIMILARITY_MATRIX.get(key, 0.5)