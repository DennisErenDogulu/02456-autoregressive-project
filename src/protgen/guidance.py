"""Sampling adapter for Person 1's frozen-backbone family classifier."""

import math
from pathlib import Path

import torch
from torch import nn

from protgen.data import FAMILIES, total_tokens
from protgen.models import GRU, FamilyClassifier, load_family_classifier


def validate_family_mapping(mapping: dict[str, int]) -> None:
    if not isinstance(mapping, dict) or set(mapping) != set(FAMILIES):
        raise ValueError(f"Classifier family mapping must contain {FAMILIES}")
    if any(type(index) is not int for index in mapping.values()) or set(mapping.values()) != set(range(len(FAMILIES))):
        raise ValueError("Classifier family indices must be a permutation of 0, 1, 2")


class PrefixFamilyClassifier(nn.Module):
    """Adapt Person 1's hidden-state head to token-prefix inference.

    Input: unpadded ``torch.long`` tokens of shape ``(batch, prefix_length)``,
    including BOS and the candidate token. Output: raw family logits of shape
    ``(batch, 3)`` on the input device, indexed by ``fam_to_idx``. The saved
    classifier backbone must be used even when the generator is another model.
    """

    def __init__(self, backbone: GRU, head: FamilyClassifier, fam_to_idx: dict[str, int]):
        super().__init__()
        validate_family_mapping(fam_to_idx)
        if (
            backbone.embedding.num_embeddings != len(total_tokens)
            or backbone.embedding.padding_idx != total_tokens.index("<PAD>")
        ):
            raise ValueError("Classifier backbone vocabulary/padding differs from the project tokenizer")
        if head.hidden_dim != backbone.gru.hidden_size or head.num_classes != len(FAMILIES):
            raise ValueError("Classifier head dimensions do not match its backbone/family mapping")
        self.backbone = backbone
        self.head = head
        self.fam_to_idx = dict(fam_to_idx)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.hidden_states(tokens)[:, -1, :])


def load_prefix_classifier(path: Path, device: torch.device) -> PrefixFamilyClassifier:
    if not path.is_file():
        raise ValueError(f"No family-classifier checkpoint found at {path}")
    backbone, head, mapping = load_family_classifier(path, device)
    return PrefixFamilyClassifier(backbone, head, mapping).eval()


def validate_guidance(strength: float, target_family: str | None, classifier) -> None:
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("guidance strength must be finite and nonnegative")
    if strength > 0:
        if target_family not in FAMILIES:
            raise ValueError(f"target_family must be one of {FAMILIES}")
        if classifier is None:
            raise ValueError("positive guidance requires a family classifier")
        validate_family_mapping(classifier.fam_to_idx)


@torch.inference_mode()
def guide_logits(
    logits: torch.Tensor,
    prefixes: torch.Tensor,
    classifier: nn.Module,
    target_family: str,
    strength: float,
    active: torch.Tensor,
) -> torch.Tensor:
    """Slide 11: add lambda * log p(family | prefix + candidate).

    The sampler supplies temperature-scaled logits with forbidden tokens masked.
    Its existing softmax normalizes the resulting weights before sampling:
    ``softmax(generator_logits / temperature + strength * log_p_family)``.
    Only eligible candidates of active sequences are scored. A zero strength
    returns the original tensor without accessing the classifier or prefixes.
    """
    validate_guidance(strength, target_family, classifier)
    if strength == 0:
        return logits
    family_index = classifier.fam_to_idx[target_family]
    rows, candidates = (torch.isfinite(logits) & active[:, None]).nonzero(as_tuple=True)
    guided = logits.clone()
    # Bound inference memory independently of the sampling batch size.
    for start in range(0, rows.numel(), 256):
        batch_rows = rows[start:start + 256]
        batch_candidates = candidates[start:start + 256]
        inputs = torch.cat((prefixes[batch_rows], batch_candidates[:, None]), dim=1)
        family_logits = classifier(inputs)
        expected_shape = (inputs.size(0), len(FAMILIES))
        if not isinstance(family_logits, torch.Tensor) or family_logits.shape != expected_shape:
            raise ValueError(f"Classifier must return family logits with shape {expected_shape}")
        if not family_logits.is_floating_point() or not torch.isfinite(family_logits).all():
            raise ValueError("Classifier must return finite floating-point logits")
        if family_logits.device != logits.device:
            raise ValueError("Classifier output must be on the sampler device")
        guided[batch_rows, batch_candidates] += strength * family_logits.log_softmax(-1)[:, family_index]
    return guided
