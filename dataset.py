"""
Dataset loading utilities for multilingual sentiment analysis.

This module provides loaders for:
1. BnSentMix (Bengali-English code-mixed sentiment)
2. IndicSentiment-Translated (Hindi sentiment)
3. Fallback synthetic/demo datasets for quick testing
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import random


class SentimentDataset(Dataset):
    """Simple sentiment dataset wrapper."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_bengali_sentiment(
    split: str = 'train',
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Load BnSentMix dataset (Bengali-English code-mixed sentiment).

    Dataset: https://huggingface.co/datasets/aplycaebous/BnSentMix
    Labels: 0=Positive, 1=Negative, 2=Neutral, 3=Mixed
    
    For binary classification, we'll map:
    - Positive (0) -> 1
    - Negative (1) -> 0
    - Neutral (2) -> 0 (treat as negative for simplicity)
    - Mixed (3) -> 1 (treat as positive for simplicity)

    Args:
        split: 'train' or 'test' (note: this dataset only has 'train', we'll split it)
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (texts, labels)
    """
    try:
        from datasets import load_dataset

        print(f"  Loading BnSentMix (Bengali-English code-mixed) dataset...")

        # Load the full dataset (only has 'train' split with 20k samples)
        dataset = load_dataset("aplycaebous/BnSentMix", split="train")
        
        # Convert to lists
        all_texts = []
        all_labels = []
        
        for example in dataset:
            text = example['Sentence']
            original_label = example['Label']  # 0=Pos, 1=Neg, 2=Neutral, 3=Mixed
            
            # Convert to binary: 0=Negative/Neutral, 1=Positive/Mixed
            if original_label == 0:  # Positive
                binary_label = 1
            elif original_label == 1:  # Negative
                binary_label = 0
            elif original_label == 2:  # Neutral
                binary_label = 0
            else:  # Mixed (3)
                binary_label = 1
            
            if text and text.strip():
                all_texts.append(text.strip())
                all_labels.append(binary_label)
        
        # Split into train/test (80/20)
        total_samples = len(all_texts)
        split_idx = int(0.8 * total_samples)
        
        if split == 'train':
            texts = all_texts[:split_idx]
            labels = all_labels[:split_idx]
        else:  # test/eval
            texts = all_texts[split_idx:]
            labels = all_labels[split_idx:]
        
        # Limit samples if requested
        if max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]

        print(f"  Loaded {len(texts)} Bengali samples ({split} split)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading BnSentMix: {e}")
        raise RuntimeError(
            f"Could not load BnSentMix dataset for Bengali. "
            f"Please ensure you have internet connection and the datasets library is installed. "
            f"Original error: {e}"
        )


def load_hindi_sentiment(
    split: str = 'train',
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Load IndicSentiment-Translated dataset (Hindi sentiment).

    Dataset: https://huggingface.co/datasets/ai4bharat/IndicSentiment-Translated
    The 'INDIC REVIEW' column contains Hindi text in Devanagari script.
    Labels: "Positive" or "Negative"

    Args:
        split: 'train' or 'test' (dataset has 'validation' and 'test')
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (texts, labels)
    """
    try:
        from datasets import load_dataset

        print(f"  Loading IndicSentiment-Translated (Hindi) dataset...")

        # Map split names (dataset has 'validation' and 'test', no 'train')
        # We'll use 'validation' for training and 'test' for evaluation
        dataset_split = 'validation' if split == 'train' else 'test'
        
        # Load dataset
        dataset = load_dataset("ai4bharat/IndicSentiment-Translated", split=dataset_split)

        texts = []
        labels = []

        for example in dataset:
            # Get Hindi text from 'INDIC REVIEW' column (contains Hindi in Devanagari)
            text = example['INDIC REVIEW']
            label_str = example['LABEL']  # "Positive" or "Negative"
            
            # Convert to binary: Positive=1, Negative=0
            if 'Positive' in label_str or 'positive' in label_str:
                label = 1
            else:
                label = 0
            
            if text and text.strip():
                texts.append(text.strip())
                labels.append(label)

            if max_samples and len(texts) >= max_samples:
                break

        print(f"  Loaded {len(texts)} Hindi samples ({split} split)")
        return texts, labels

    except Exception as e:
        print(f"  Error loading IndicSentiment-Translated: {e}")
        raise RuntimeError(
            f"Could not load IndicSentiment-Translated dataset for Hindi. "
            f"Please ensure you have internet connection and the datasets library is installed. "
            f"Original error: {e}"
        )


def load_sentiment_data(
    language: str,
    split: str = 'train',
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Load sentiment data for the specified language.

    Args:
        language: Language code ('bengali', 'hindi', 'bn', or 'hi')
        split: 'train' or 'test'
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (texts, labels)
    """
    # Normalize language name
    lang_map = {
        'bengali': 'bengali',
        'hindi': 'hindi',
        'bn': 'bengali',
        'hi': 'hindi'
    }
    
    lang = lang_map.get(language.lower())
    if not lang:
        raise ValueError(f"Unsupported language: {language}. Supported: bengali, hindi, bn, hi")
    
    # Load appropriate dataset
    if lang == 'bengali':
        return load_bengali_sentiment(split, max_samples)
    elif lang == 'hindi':
        return load_hindi_sentiment(split, max_samples)
    else:
        raise ValueError(f"Unknown language: {lang}")


def _generate_demo_data(
    language: str,
    split: str = 'train',
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Generate simple demo data for testing the pipeline.

    This creates synthetic bilingual data with language-specific vocabulary
    to simulate the actual task. In production, replace with real data.

    Args:
        language: 'bengali' or 'hindi'
        split: 'train' or 'test'
        max_samples: Number of samples to generate

    Returns:
        Tuple of (texts, labels)
    """
    random.seed(42 if split == 'train' else 43)

    # Simple vocabulary for demo purposes
    vocab_templates = {
        'bengali': {
            'positive': [
                "এটি একটি চমৎকার সিনেমা",  # This is an excellent movie
                "আমি এই বইটি খুব পছন্দ করেছি",  # I really liked this book
                "দুর্দান্ত পরিষেবা",  # Great service
                "অসাধারণ অভিজ্ঞতা",  # Amazing experience
                "খুবই ভালো লেগেছে",  # Liked it very much
            ],
            'negative': [
                "এটি খুবই খারাপ ছিল",  # This was very bad
                "ভয়ঙ্কর অভিজ্ঞতা",  # Terrible experience
                "সময়ের অপচয়",  # Waste of time
                "খুবই নিম্নমানের",  # Very low quality
                "একদম পছন্দ হয়নি",  # Did not like at all
            ]
        },
        'hindi': {
            'positive': [
                "यह एक शानदार फिल्म है",  # This is a great movie
                "मुझे यह किताब बहुत पसंद आई",  # I really liked this book
                "बहुत अच्छी सेवा",  # Very good service
                "शानदार अनुभव",  # Great experience
                "बेहतरीन था",  # Was excellent
            ],
            'negative': [
                "यह बहुत बुरा था",  # This was very bad
                "भयानक अनुभव",  # Terrible experience
                "समय की बर्बादी",  # Waste of time
                "बहुत खराब गुणवत्ता",  # Very bad quality
                "बिल्कुल पसंद नहीं आया",  # Did not like at all
            ]
        }
    }

    if language not in vocab_templates:
        raise ValueError(f"Unsupported language: {language}")

    # Generate data
    texts = []
    labels = []

    n_samples = max_samples if max_samples else 1000
    pos_templates = vocab_templates[language]['positive']
    neg_templates = vocab_templates[language]['negative']

    for i in range(n_samples):
        if i % 2 == 0:
            # Positive sample
            text = random.choice(pos_templates)
            label = 1
        else:
            # Negative sample
            text = random.choice(neg_templates)
            label = 0

        # Add some variation with random suffixes/prefixes
        if random.random() < 0.3:
            text = text + " " + random.choice(pos_templates if label == 1 else neg_templates)

        texts.append(text)
        labels.append(label)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), list(labels)


def prepare_dataloaders(
    languages: List[str],
    tokenizer: AutoTokenizer,
    config,
    use_demo_data: bool = False
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Prepare DataLoaders for all languages.

    Args:
        languages: List of language codes ('bengali', 'hindi', 'bn', or 'hi')
        tokenizer: Tokenizer to use
        config: Experiment configuration (must have: train_size, eval_size, max_length, batch_size)
        use_demo_data: Whether to use demo data (for testing)

    Returns:
        Nested dict: {language: {'train': DataLoader, 'eval': DataLoader}}
    """
    dataloaders = {}

    for language in languages:
        print(f"\nLoading data for {language}...")

        if use_demo_data:
            # Use demo data
            train_texts, train_labels = _generate_demo_data(
                language, 'train', config.train_size
            )
            eval_texts, eval_labels = _generate_demo_data(
                language, 'test', config.eval_size
            )
        else:
            # Load real data
            train_texts, train_labels = load_sentiment_data(
                language, 'train', config.train_size
            )
            eval_texts, eval_labels = load_sentiment_data(
                language, 'test', config.eval_size
            )

        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, tokenizer, config.max_length
        )
        eval_dataset = SentimentDataset(
            eval_texts, eval_labels, tokenizer, config.max_length
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        dataloaders[language] = {
            'train': train_loader,
            'eval': eval_loader
        }

        print(f"  ✓ {language}: {len(train_dataset)} train, {len(eval_dataset)} eval samples")

    return dataloaders