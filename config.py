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
    batch_size: int = 8  # Bigger batches
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # EWC configuration
    ewc_lambda: float = 5000.0
    fisher_sample_size: int = 1000  # More Fisher samples

    # Linguistic similarity configuration
    bengali_hindi_similarity: float = 0.6

    # Experimental conditions
    methods: list = None

    # Data configuration
    train_size: int = None  # None = use ALL available data
    eval_size: int = None   # None = use ALL available data
    random_seed: int = 42

    # Output configuration
    output_dir: str = "./results"
    save_models: bool = False

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['naive', 'ewc', 'ling_ewc', 'random_ewc']


LANGUAGE_SIMILARITY_MATRIX = {
    ('bengali', 'hindi'): 0.6,
    ('hindi', 'bengali'): 0.6,
    ('bengali', 'bengali'): 1.0,
    ('hindi', 'hindi'): 1.0,
}


def get_linguistic_similarity(lang1: str, lang2: str) -> float:
    """Get linguistic similarity score between two languages."""
    key = (lang1.lower(), lang2.lower())
    return LANGUAGE_SIMILARITY_MATRIX.get(key, 0.5)