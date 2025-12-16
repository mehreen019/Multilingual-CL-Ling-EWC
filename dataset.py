"""
Dataset loading utilities for multilingual sentiment analysis.

This module provides loaders for:
1. IndicNLP Sentiment Corpus (if available)
2. Fallback synthetic/demo datasets for quick testing
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


def load_indicnlp_sentiment(
    language: str,
    split: str = 'train',
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Load IndicSentiment dataset from AI4Bharat.

    Dataset: https://huggingface.co/datasets/ai4bharat/IndicSentiment
    Languages: Bengali (bn), Hindi (hi), and others
    Labels: 0 (negative), 1 (neutral), 2 (positive)

    Args:
        language: Language code ('bengali' or 'hindi')
        split: 'train' or 'test'
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (texts, labels)
    """
    try:
        from datasets import load_dataset

        # Map language names to ISO codes
        lang_map = {
            'bengali': 'bn',
            'hindi': 'hi',
            'bn': 'bn',
            'hi': 'hi'
        }

        lang_code = lang_map.get(language.lower())
        if not lang_code:
            raise ValueError(f"Unsupported language: {language}")

        print(f"  Loading ai4bharat/IndicSentiment for {language} ({lang_code})...")

        # Load dataset
        dataset = load_dataset("ai4bharat/IndicSentiment", lang_code, split=split, trust_remote_code=True)

        # Extract texts and labels
        texts = []
        labels = []

        for example in dataset:
            # IndicSentiment has 'INDIC REVIEW' field for text
            # and 'LABEL' field with values: positive, negative, neutral
            text = example.get('INDIC REVIEW', example.get('text', ''))
            label_str = example.get('LABEL', example.get('label', 'neutral'))

            # Convert label to integer
            # We'll use binary for simplicity: positive=1, negative/neutral=0
            if isinstance(label_str, str):
                if 'positive' in label_str.lower():
                    label = 1
                else:
                    label = 0  # Treat neutral and negative as same class
            else:
                label = int(label_str)

            if text and text.strip():  # Only add non-empty texts
                texts.append(text.strip())
                labels.append(label)

            if max_samples and len(texts) >= max_samples:
                break

        print(f"  Loaded {len(texts)} samples for {language}")
        return texts, labels

    except Exception as e:
        print(f"  Error loading IndicSentiment: {e}")
        print(f"  Attempting fallback strategies...")

        # Fallback: Try alternative dataset names
        alternative_datasets = [
            ("mteb/bengali_sentiment_analysis", 'bengali'),
            ("bigscience-data/roots_indic-bn_bangla_sentiment_classification_datasets", 'bengali'),
        ]

        for dataset_name, supported_lang in alternative_datasets:
            if language.lower() not in supported_lang.lower():
                continue

            try:
                print(f"  Trying {dataset_name}...")
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

                texts = []
                labels = []

                for example in dataset:
                    # Try common field names
                    text = example.get('text', example.get('sentence', example.get('review', '')))
                    label = example.get('label', example.get('sentiment', 0))

                    if isinstance(label, str):
                        label = 1 if 'pos' in label.lower() else 0

                    if text and text.strip():
                        texts.append(text.strip())
                        labels.append(int(label))

                    if max_samples and len(texts) >= max_samples:
                        break

                if texts:
                    print(f"  Successfully loaded {len(texts)} samples from {dataset_name}")
                    return texts, labels

            except Exception as e2:
                print(f"  Failed to load {dataset_name}: {e2}")
                continue

        # If all else fails, raise error
        raise RuntimeError(
            f"Could not load any dataset for {language}. "
            f"Please ensure you have internet connection and the datasets library is installed. "
            f"Original error: {e}"
        )


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
    # In real scenario, these would be actual Bengali/Hindi sentences
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
        languages: List of language codes
        tokenizer: Tokenizer to use
        config: Experiment configuration
        use_demo_data: Whether to use demo data (for testing)

    Returns:
        Nested dict: {language: {'train': DataLoader, 'eval': DataLoader}}
    """
    dataloaders = {}

    for language in languages:
        print(f"Loading data for {language}...")

        if use_demo_data:
            # Use demo data
            train_texts, train_labels = _generate_demo_data(
                language, 'train', config.train_size
            )
            eval_texts, eval_labels = _generate_demo_data(
                language, 'test', config.eval_size
            )
        else:
            # Load real data from IndicSentiment or fallbacks
            train_texts, train_labels = load_indicnlp_sentiment(
                language, 'train', config.train_size
            )
            eval_texts, eval_labels = load_indicnlp_sentiment(
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

        print(f"  {language}: {len(train_dataset)} train, {len(eval_dataset)} eval samples")

    return dataloaders
