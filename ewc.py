"""
Implementation of Elastic Weight Consolidation (EWC) and its linguistic variant.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from copy import deepcopy
from tqdm import tqdm


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    This implementation computes Fisher Information Matrix diagonal
    and applies EWC penalty during training on subsequent tasks.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize EWC.

        Args:
            model: The model to apply EWC to
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.previous_tasks = []  # List of (params, fisher, language) tuples

    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        language: str,
        sample_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal for current model parameters.

        The Fisher Information is computed as the expected value of the squared
        gradients of the log-likelihood with respect to the model parameters.

        Args:
            dataloader: DataLoader for the current task
            language: Language identifier for this task
            sample_size: Number of samples to use (None = use all)

        Returns:
            Dictionary mapping parameter names to Fisher diagonal values
        """
        self.model.eval()
        fisher_dict = {}

        # Initialize Fisher values to zero
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # Compute Fisher Information
        samples_processed = 0
        for batch in tqdm(dataloader, desc=f"Computing Fisher for {language}"):
            if sample_size and samples_processed >= sample_size:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2

            samples_processed += input_ids.size(0)

        # Normalize by number of samples
        num_samples = samples_processed
        for name in fisher_dict:
            fisher_dict[name] /= num_samples

        self.model.train()
        return fisher_dict

    def save_task(
        self,
        dataloader: DataLoader,
        language: str,
        sample_size: Optional[int] = None
    ):
        """
        Save current task parameters and Fisher Information.

        Args:
            dataloader: DataLoader for computing Fisher
            language: Language identifier
            sample_size: Number of samples for Fisher computation
        """
        # Save current model parameters
        params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Compute Fisher Information
        fisher = self.compute_fisher_information(dataloader, language, sample_size)

        # Store task information
        self.previous_tasks.append({
            'params': params,
            'fisher': fisher,
            'language': language
        })

        print(f"Saved task: {language} (Total tasks: {len(self.previous_tasks)})")

    def compute_ewc_loss(
        self,
        ewc_lambda: float = 5000.0,
        similarity_scale: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute EWC penalty loss.

        The EWC loss penalizes changes to important parameters:
        L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2

        where F_i is the Fisher Information and θ_i* are previous parameters.

        Args:
            ewc_lambda: Strength of EWC penalty
            similarity_scale: Optional dict mapping language to similarity scaling factor

        Returns:
            EWC penalty loss
        """
        ewc_loss = torch.tensor(0.0, device=self.device)

        for task_idx, task in enumerate(self.previous_tasks):
            task_loss = torch.tensor(0.0, device=self.device)

            # Get similarity scaling factor for this task
            scale = 1.0
            if similarity_scale is not None:
                task_language = task['language']
                scale = similarity_scale.get(task_language, 1.0)

            # Compute penalty for this task
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in task['fisher']:
                    fisher = task['fisher'][name]
                    old_param = task['params'][name]

                    # EWC penalty: (λ * similarity_scale / 2) * F * (θ - θ*)^2
                    task_loss += (fisher * (param - old_param) ** 2).sum()

            # Apply similarity scaling and lambda
            ewc_loss += (scale * ewc_lambda / 2.0) * task_loss

        return ewc_loss


class LinguisticEWC(EWC):
    """
    Linguistically-aware EWC that scales penalty based on language similarity.

    The key innovation: when learning language L_new, the EWC penalty for
    previous language L_old is scaled by similarity(L_old, L_new).

    Higher similarity → weaker penalty (allow more parameter changes, transfer is beneficial)
    Lower similarity → stronger penalty (protect parameters more, prevent forgetting)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)

    def compute_linguistic_ewc_loss(
        self,
        current_language: str,
        ewc_lambda: float = 5000.0,
        similarity_fn: Optional[callable] = None,
        invert_similarity: bool = True
    ) -> torch.Tensor:
        """
        Compute linguistically-scaled EWC loss.

        Args:
            current_language: Language currently being learned
            ewc_lambda: Base EWC penalty strength
            similarity_fn: Function to compute similarity(prev_lang, curr_lang)
            invert_similarity: If True, use (1 - similarity) as scale
                             (higher similarity = lower penalty)

        Returns:
            Linguistically-scaled EWC loss
        """
        if similarity_fn is None:
            from config import get_linguistic_similarity
            similarity_fn = get_linguistic_similarity

        # Compute similarity scaling for each previous task
        similarity_scales = {}
        for task in self.previous_tasks:
            prev_language = task['language']
            similarity = similarity_fn(prev_language, current_language)

            # Key insight: if languages are similar, we want LESS penalty
            # because transfer is beneficial and we want to allow parameter updates
            if invert_similarity:
                # Scale = (1 - similarity): high similarity → low penalty
                scale = 1.0 - similarity
            else:
                # Scale = similarity: high similarity → high penalty
                scale = similarity

            similarity_scales[prev_language] = scale
            print(f"  Similarity({prev_language}->{current_language}) = {similarity:.3f}, "
                  f"EWC scale = {scale:.3f}")

        # Compute EWC loss with similarity scaling
        return self.compute_ewc_loss(ewc_lambda, similarity_scales)
