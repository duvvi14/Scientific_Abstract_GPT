
"""
Common GPT components for the Scientific Abstract GPT project.

This module contains:

1. GPT configuration
2. Reproducibility utilities
3. Masked self-attention
4. Multi-head attention
5. Feed-forward network
6. Transformer blocks
7. Decoder-only GPT language model
8. Tokenizer loading
9. Prompt creation
10. Autoregressive text generation
11. Abstract extraction
12. Model checkpoint loading
13. Parameter counting
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(
    seed: int = 42,
    deterministic: bool = False
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Random seed value.

    deterministic:
        When True, PyTorch attempts to use deterministic
        algorithms. This can reduce performance and may not
        be supported by every GPU operation.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True
            )
        except Exception:
            pass

    else:

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ============================================================
# 2. GPT Configuration
# ============================================================

@dataclass
class GPTConfig:
    """
    Configuration for the decoder-only GPT model.
    """

    vocab_size: int
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration values.
        """

        if self.vocab_size <= 0:

            raise ValueError(
                "vocab_size must be greater than zero."
            )

        if self.block_size <= 0:

            raise ValueError(
                "block_size must be greater than zero."
            )

        if self.n_embd <= 0:

            raise ValueError(
                "n_embd must be greater than zero."
            )

        if self.n_head <= 0:

            raise ValueError(
                "n_head must be greater than zero."
            )

        if self.n_layer <= 0:

            raise ValueError(
                "n_layer must be greater than zero."
            )

        if self.n_embd % self.n_head != 0:

            raise ValueError(
                "n_embd must be divisible by n_head. "
                f"Received n_embd={self.n_embd} and "
                f"n_head={self.n_head}."
            )

        if not 0.0 <= self.dropout < 1.0:

            raise ValueError(
                "dropout must be between 0 and 1."
            )

    @property
    def head_size(self) -> int:
        """
        Embedding dimension handled by one attention head.
        """

        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        """

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        config_dictionary: Dict[str, Any]
    ) -> "GPTConfig":
        """
        Create GPTConfig from a dictionary.

        Extra checkpoint keys are ignored.
        """

        valid_keys = {
            "vocab_size",
            "block_size",
            "n_embd",
            "n_head",
            "n_layer",
            "dropout",
            "bias"
        }

        filtered_config = {
            key: value
            for key, value
            in config_dictionary.items()
            if key in valid_keys
        }

        return cls(
            **filtered_config
        )


# ============================================================
# 3. Single Masked Self-Attention Head
# ============================================================

class CausalSelfAttentionHead(nn.Module):
    """
    One masked self-attention head.

    The causal mask ensures that a token can attend only to
    itself and previous tokens, not future tokens.
    """

    def __init__(
        self,
        config: GPTConfig,
        head_size: Optional[int] = None
    ) -> None:

        super().__init__()

        self.config = config

        self.head_size = (
            head_size
            if head_size is not None
            else config.head_size
        )

        self.key = nn.Linear(
            config.n_embd,
            self.head_size,
            bias=config.bias
        )

        self.query = nn.Linear(
            config.n_embd,
            self.head_size,
            bias=config.bias
        )

        self.value = nn.Linear(
            config.n_embd,
            self.head_size,
            bias=config.bias
        )

        self.attention_dropout = nn.Dropout(
            config.dropout
        )

        causal_mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply masked scaled dot-product attention.

        Parameters
        ----------
        x:
            Tensor with shape:
            [batch_size, sequence_length, n_embd]

        Returns
        -------
        Tensor with shape:
        [batch_size, sequence_length, head_size]
        """

        batch_size, sequence_length, _ = x.shape

        if sequence_length > self.config.block_size:

            raise ValueError(
                "Input sequence length exceeds block_size. "
                f"Received {sequence_length}, but block_size "
                f"is {self.config.block_size}."
            )

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attention_scores = (
            queries @ keys.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores /
            (self.head_size ** 0.5)
        )

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                float("-inf")
            )
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = (
            self.attention_dropout(
                attention_weights
            )
        )

        output = (
            attention_weights @ values
        )

        return output


# ============================================================
# 4. Multi-Head Masked Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multiple masked self-attention heads operating in parallel.
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        self.config = config

        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                config=config,
                head_size=config.head_size
            )
            for _ in range(
                config.n_head
            )
        ])

        self.output_projection = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.residual_dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply all attention heads and combine their outputs.
        """

        multi_head_output = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        projected_output = (
            self.output_projection(
                multi_head_output
            )
        )

        return self.residual_dropout(
            projected_output
        )


# ============================================================
# 5. Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward neural network.

    GPT-style feed-forward networks expand the embedding
    dimension by a factor of four and then project it back.
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        hidden_size = (
            4 * config.n_embd
        )

        self.network = nn.Sequential(
            nn.Linear(
                config.n_embd,
                hidden_size,
                bias=config.bias
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                config.n_embd,
                bias=config.bias
            ),
            nn.Dropout(
                config.dropout
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.network(x)


# ============================================================
# 6. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    One decoder-only Transformer block.

    The block contains:

    1. Layer normalization
    2. Masked multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. Feed-forward network
    6. Residual connection
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        self.layer_norm_attention = nn.LayerNorm(
            config.n_embd
        )

        self.self_attention = (
            MultiHeadAttention(
                config
            )
        )

        self.layer_norm_feed_forward = (
            nn.LayerNorm(
                config.n_embd
            )
        )

        self.feed_forward = (
            FeedForward(
                config
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = (
            x +
            self.self_attention(
                self.layer_norm_attention(
                    x
                )
            )
        )

        x = (
            x +
            self.feed_forward(
                self.layer_norm_feed_forward(
                    x
                )
            )
        )

        return x


# ============================================================
# 7. GPT Language Model
# ============================================================

class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT language model.

    The model predicts the next token for each position in
    the input sequence.
    """

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config
                )
                for _ in range(
                    config.n_layer
                )
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.n_embd
        )

        self.language_model_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying:
        # The input token embedding and output token
        # projection share the same weight matrix.
        self.language_model_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._initialize_weights
        )

        self._initialize_residual_projections()

    def _initialize_weights(
        self,
        module: nn.Module
    ) -> None:
        """
        Initialize model weights.
        """

        if isinstance(
            module,
            nn.Linear
        ):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def _initialize_residual_projections(
        self
    ) -> None:
        """
        Apply scaled initialization to residual projections.

        This improves stability in deeper Transformer models.
        """

        residual_scale = (
            0.02 /
            ((2 * self.config.n_layer) ** 0.5)
        )

        for module_name, parameter in (
            self.named_parameters()
        ):

            if (
                module_name.endswith(
                    "output_projection.weight"
                )
                or module_name.endswith(
                    "network.2.weight"
                )
            ):

                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=residual_scale
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[
            torch.Tensor
        ] = None
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor]
    ]:
        """
        Perform a forward pass.

        Parameters
        ----------
        input_ids:
            Token IDs with shape:
            [batch_size, sequence_length]

        targets:
            Next-token target IDs with shape:
            [batch_size, sequence_length]

        Returns
        -------
        logits:
            Prediction scores with shape:
            [batch_size, sequence_length, vocab_size]

        loss:
            Cross-entropy loss when targets are provided.
            Otherwise, None.
        """

        if input_ids.ndim != 2:

            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]."
            )

        batch_size, sequence_length = (
            input_ids.shape
        )

        if sequence_length > self.config.block_size:

            raise ValueError(
                "Input sequence length exceeds block_size. "
                f"Received {sequence_length}, but block_size "
                f"is {self.config.block_size}."
            )

        positions = torch.arange(
            start=0,
            end=sequence_length,
            device=input_ids.device,
            dtype=torch.long
        )

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = (
            token_embeddings +
            position_embeddings
        )

        x = self.embedding_dropout(x)

        x = self.transformer_blocks(x)

        x = self.final_layer_norm(x)

        logits = self.language_model_head(
            x
        )

        loss = None

        if targets is not None:

            if targets.shape != input_ids.shape:

                raise ValueError(
                    "targets must have the same shape "
                    "as input_ids."
                )

            flattened_logits = logits.reshape(
                batch_size * sequence_length,
                self.config.vocab_size
            )

            flattened_targets = targets.reshape(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                flattened_logits,
                flattened_targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        end_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Parameters
        ----------
        input_ids:
            Initial prompt token IDs with shape
            [batch_size, prompt_length].

        max_new_tokens:
            Maximum number of tokens to generate.

        temperature:
            Sampling temperature. Lower values make generation
            more deterministic.

        top_k:
            Keep only the top-k most probable tokens.
            Set to None or 0 to disable.

        top_p:
            Nucleus-sampling probability threshold.
            Set to None to disable.

        repetition_penalty:
            Values greater than 1 reduce repeated-token
            probabilities.

        end_token_id:
            Stop generation when this token is produced.
        """

        if input_ids.ndim != 2:

            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]."
            )

        if input_ids.size(1) == 0:

            raise ValueError(
                "The generation prompt cannot be empty."
            )

        if max_new_tokens < 0:

            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if temperature <= 0:

            raise ValueError(
                "temperature must be greater than zero."
            )

        if repetition_penalty < 1.0:

            raise ValueError(
                "repetition_penalty must be at least 1.0."
            )

        if (
            top_p is not None
            and not 0.0 < top_p <= 1.0
        ):

            raise ValueError(
                "top_p must be between 0 and 1."
            )

        generated_ids = input_ids

        for _ in range(
            max_new_tokens
        ):

            model_context = generated_ids[
                :,
                -self.config.block_size:
            ]

            logits, _ = self(
                model_context
            )

            next_token_logits = logits[
                :,
                -1,
                :
            ].clone()

            if repetition_penalty > 1.0:

                for batch_index in range(
                    generated_ids.size(0)
                ):

                    previous_tokens = torch.unique(
                        generated_ids[
                            batch_index
                        ]
                    )

                    previous_logits = (
                        next_token_logits[
                            batch_index,
                            previous_tokens
                        ]
                    )

                    penalized_logits = torch.where(
                        previous_logits < 0,
                        previous_logits *
                        repetition_penalty,
                        previous_logits /
                        repetition_penalty
                    )

                    next_token_logits[
                        batch_index,
                        previous_tokens
                    ] = penalized_logits

            next_token_logits = (
                next_token_logits /
                temperature
            )

            if (
                top_k is not None
                and top_k > 0
            ):

                effective_top_k = min(
                    int(top_k),
                    next_token_logits.size(-1)
                )

                top_values, _ = torch.topk(
                    next_token_logits,
                    effective_top_k,
                    dim=-1
                )

                top_k_cutoff = top_values[
                    :,
                    -1
                ].unsqueeze(-1)

                next_token_logits = (
                    next_token_logits.masked_fill(
                        next_token_logits
                        < top_k_cutoff,
                        float("-inf")
                    )
                )

            if (
                top_p is not None
                and top_p < 1.0
            ):

                sorted_logits, sorted_indices = (
                    torch.sort(
                        next_token_logits,
                        descending=True,
                        dim=-1
                    )
                )

                sorted_probabilities = F.softmax(
                    sorted_logits,
                    dim=-1
                )

                cumulative_probabilities = (
                    torch.cumsum(
                        sorted_probabilities,
                        dim=-1
                    )
                )

                sorted_remove_mask = (
                    cumulative_probabilities
                    > top_p
                )

                sorted_remove_mask[
                    :,
                    1:
                ] = sorted_remove_mask[
                    :,
                    :-1
                ].clone()

                sorted_remove_mask[
                    :,
                    0
                ] = False

                remove_mask = torch.zeros_like(
                    sorted_remove_mask
                )

                remove_mask.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_remove_mask
                )

                next_token_logits = (
                    next_token_logits.masked_fill(
                        remove_mask,
                        float("-inf")
                    )
                )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1
            )

            if torch.isnan(
                probabilities
            ).any():

                raise RuntimeError(
                    "NaN values appeared in generation "
                    "probabilities."
                )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token
                ],
                dim=1
            )

            if end_token_id is not None:

                if torch.all(
                    next_token.squeeze(-1)
                    == end_token_id
                ):

                    break

        return generated_ids


# ============================================================
# 8. Parameter Counting
# ============================================================

def count_parameters(
    model: nn.Module,
    trainable_only: bool = True
) -> int:
    """
    Count model parameters.

    Parameters
    ----------
    model:
        PyTorch model.

    trainable_only:
        Count only parameters with requires_grad=True.
    """

    if trainable_only:

        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# 9. Tokenizer Loading
# ============================================================

def load_tokenizer(
    tokenizer_path: str
) -> Tokenizer:
    """
    Load a saved Hugging Face Tokenizers tokenizer.

    Parameters
    ----------
    tokenizer_path:
        Path to tokenizer.json.
    """

    if not isinstance(
        tokenizer_path,
        str
    ):

        raise TypeError(
            "tokenizer_path must be a string."
        )

    import os

    if not os.path.exists(
        tokenizer_path
    ):

        raise FileNotFoundError(
            "Tokenizer file was not found: "
            f"{tokenizer_path}"
        )

    tokenizer = Tokenizer.from_file(
        tokenizer_path
    )

    if tokenizer.get_vocab_size() <= 0:

        raise ValueError(
            "Loaded tokenizer has an empty vocabulary."
        )

    return tokenizer


# ============================================================
# 10. Prompt Creation
# ============================================================

def _clean_prompt_value(
    value: Any
) -> str:
    """
    Normalize a title or subject used in a prompt.
    """

    if value is None:
        return ""

    return " ".join(
        str(value).strip().split()
    )


def create_prompt(
    title: str,
    subject: str = "Machine Learning"
) -> str:
    """
    Create a structured prompt for abstract generation.

    Example
    -------
    <TITLE> Deep Learning for Medical Imaging
    <SUBJECT> Machine Learning
    <ABSTRACT>
    """

    cleaned_title = _clean_prompt_value(
        title
    )

    cleaned_subject = _clean_prompt_value(
        subject
    )

    if not cleaned_title:

        raise ValueError(
            "A research-paper title is required."
        )

    if not cleaned_subject:

        cleaned_subject = (
            "Machine Learning"
        )

    return (
        f"<TITLE> {cleaned_title} "
        f"<SUBJECT> {cleaned_subject} "
        f"<ABSTRACT>"
    )


# ============================================================
# 11. Text Generation Utility
# ============================================================

@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: Optional[
        Union[str, torch.device]
    ] = None,
    max_new_tokens: int = 250,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.1,
    end_token: str = "<END>",
    skip_special_tokens: bool = False
) -> str:
    """
    Tokenize a prompt, generate tokens, and decode the result.
    """

    if not isinstance(
        model,
        GPTLanguageModel
    ):

        raise TypeError(
            "model must be a GPTLanguageModel instance."
        )

    if not isinstance(
        tokenizer,
        Tokenizer
    ):

        raise TypeError(
            "tokenizer must be a Tokenizer instance."
        )

    if not isinstance(
        prompt,
        str
    ) or not prompt.strip():

        raise ValueError(
            "prompt must be a non-empty string."
        )

    if device is None:

        device = next(
            model.parameters()
        ).device

    device = torch.device(
        device
    )

    prompt_encoding = tokenizer.encode(
        prompt
    )

    prompt_ids = prompt_encoding.ids

    if len(prompt_ids) == 0:

        raise ValueError(
            "The tokenizer produced an empty prompt."
        )

    prompt_tensor = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    end_token_id = tokenizer.token_to_id(
        end_token
    )

    model.eval()

    generated_ids = model.generate(
        input_ids=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        end_token_id=end_token_id
    )

    decoded_text = tokenizer.decode(
        generated_ids[
            0
        ].detach().cpu().tolist(),
        skip_special_tokens=skip_special_tokens
    )

    return decoded_text


# ============================================================
# 12. Abstract Extraction
# ============================================================

def extract_abstract(
    generated_text: str,
    abstract_token: str = "<ABSTRACT>",
    end_token: str = "<END>"
) -> str:
    """
    Extract only the generated abstract from complete model text.
    """

    if generated_text is None:

        return ""

    text = str(
        generated_text
    )

    if abstract_token in text:

        text = text.split(
            abstract_token,
            1
        )[1]

    if end_token in text:

        text = text.split(
            end_token,
            1
        )[0]

    text = " ".join(
        text.strip().split()
    )

    return text


# ============================================================
# 13. Checkpoint Configuration Extraction
# ============================================================

def _extract_checkpoint_config(
    checkpoint: Dict[str, Any]
) -> GPTConfig:
    """
    Extract GPTConfig from a model checkpoint.
    """

    possible_config_keys = [
        "config",
        "model_config",
        "gpt_config"
    ]

    config_dictionary = None

    for key in possible_config_keys:

        if key in checkpoint:

            config_dictionary = (
                checkpoint[key]
            )

            break

    if config_dictionary is None:

        raise KeyError(
            "Checkpoint does not contain a model "
            "configuration. Expected one of: "
            f"{possible_config_keys}"
        )

    if isinstance(
        config_dictionary,
        GPTConfig
    ):

        return config_dictionary

    if not isinstance(
        config_dictionary,
        dict
    ):

        raise TypeError(
            "Checkpoint configuration must be "
            "a dictionary or GPTConfig."
        )

    return GPTConfig.from_dict(
        config_dictionary
    )


# ============================================================
# 14. Model Checkpoint Loading
# ============================================================

def load_model_checkpoint(
    checkpoint_path: str,
    device: Optional[
        Union[str, torch.device]
    ] = None,
    evaluation_mode: bool = True,
    strict: bool = True
) -> Tuple[
    GPTLanguageModel,
    GPTConfig,
    Dict[str, Any]
]:
    """
    Load a trained GPT model from a checkpoint.

    Supported checkpoint formats
    ----------------------------
    Recommended format:

    {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "validation_loss": validation_loss
    }

    Also supports:

    {
        "state_dict": model.state_dict(),
        "config": ...
    }

    Returns
    -------
    model:
        Loaded GPTLanguageModel.

    config:
        GPTConfig used to create the model.

    checkpoint:
        Complete checkpoint dictionary.
    """

    import os

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "Checkpoint was not found: "
            f"{checkpoint_path}"
        )

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    config = _extract_checkpoint_config(
        checkpoint
    )

    model = GPTLanguageModel(
        config
    ).to(device)

    if "model_state_dict" in checkpoint:

        model_state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        model_state_dict = checkpoint[
            "state_dict"
        ]

    else:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict' or 'state_dict'."
        )

    model.load_state_dict(
        model_state_dict,
        strict=strict
    )

    if evaluation_mode:

        model.eval()

    return (
        model,
        config,
        checkpoint
    )
