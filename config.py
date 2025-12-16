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
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # EWC configuration
    ewc_lambda: float = 5000.0  # Standard EWC penalty strength
    fisher_sample_size: int = 1000  # Number of samples for Fisher estimation

    # Linguistic similarity configuration
    # Bengali-Hindi similarity score (based on linguistic features)
    # Both are Indo-Aryan, SOV word order, but different scripts
    # Moderate similarity: ~0.6 (script differs, but syntax/morphology similar)
    bengali_hindi_similarity: float = 0.6

    # Experimental conditions
    methods: list = None  # Will be set to ['naive', 'ewc', 'ling_ewc', 'random_ewc']

    # Data configuration
    train_size: int = 1000  # Samples per language for training
    eval_size: int = 200    # Samples per language for evaluation
    random_seed: int = 42

    # Output configuration
    output_dir: str = "./results"
    save_models: bool = True

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['naive', 'ewc', 'ling_ewc', 'random_ewc']


# Linguistic feature-based similarity computation
# For the PoC, we use a simplified similarity matrix based on known linguistic features
LANGUAGE_SIMILARITY_MATRIX = {
    ('bengali', 'hindi'): 0.6,   # Moderate similarity (Indo-Aryan, SOV, different scripts)
    ('hindi', 'bengali'): 0.6,   # Symmetric
    ('bengali', 'bengali'): 1.0, # Self-similarity
    ('hindi', 'hindi'): 1.0,     # Self-similarity
}

# Linguistic features used for similarity computation (for documentation)
# 1. Script similarity: Different (Bengali vs Devanagari) -> Low (0.2)
# 2. Word order: Both SOV -> High (1.0)
# 3. Morphology: Both highly inflected, similar case systems -> High (0.8)
# 4. Language family: Both Indo-Aryan -> High (1.0)
# 5. Lexical overlap: Moderate due to shared Sanskrit roots -> Medium (0.6)
# Average: (0.2 + 1.0 + 0.8 + 1.0 + 0.6) / 5 = 0.72, conservatively set to 0.6


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
    return LANGUAGE_SIMILARITY_MATRIX.get(key, 0.5)  # Default to 0.5 if unknown
