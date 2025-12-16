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