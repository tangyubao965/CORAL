
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class CORALOutput:
    """Container returned by CORALModel.forward."""

    loss: torch.Tensor
    logits: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    gate_loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    gate_probs: Optional[torch.Tensor] = None


class ViewAwareLoRALinear(nn.Module):
    """A frozen linear layer augmented with gated global/local LoRA residuals.

    Given an original linear transformation W x, this module computes

        W x + gamma_g(x) * B_g A_g x + gamma_l(x) * B_l A_l x,

    where gamma_g and gamma_l are predicted by a lightweight gating network.
    The original linear layer is kept as the shared decoding backbone, while the
    low-rank residuals specialize decoding for different docid views.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("ViewAwareLoRALinear can only wrap nn.Linear modules.")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        # Global-view LoRA branch.
        self.global_A = nn.Linear(self.in_features, rank, bias=False)
        self.global_B = nn.Linear(rank, self.out_features, bias=False)

        # Local-view LoRA branch.
        self.local_A = nn.Linear(self.in_features, rank, bias=False)
        self.local_B = nn.Linear(rank, self.out_features, bias=False)

        # Per-token gating. For x with shape [B, T, D], this returns [B, T, 2].
        self.gate = nn.Linear(self.in_features, 2, bias=True)

        self.last_gate_probs: Optional[torch.Tensor] = None

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # Standard LoRA initialization: A is random, B is zero, so the injected
        # module initially behaves exactly like the original linear layer.
        nn.init.kaiming_uniform_(self.global_A.weight, a=5**0.5)
        nn.init.zeros_(self.global_B.weight)
        nn.init.kaiming_uniform_(self.local_A.weight, a=5**0.5)
        nn.init.zeros_(self.local_B.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        dropped = self.dropout(x)
        global_delta = self.global_B(self.global_A(dropped)) * self.scaling
        local_delta = self.local_B(self.local_A(dropped)) * self.scaling

        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        self.last_gate_probs = gate_probs

        global_weight = gate_probs[..., 0:1]
        local_weight = gate_probs[..., 1:2]
        return base_out + global_weight * global_delta + local_weight * local_delta


class CORALModel(nn.Module):
    """CORAL model with hierarchical multi-view docid decoding.

    Parameters
    ----------
    model_name:
        HuggingFace T5 checkpoint name.
    lora_rank:
        Rank of the global/local LoRA branches.
    lora_alpha:
        Scaling factor for LoRA residuals.
    lora_dropout:
        Dropout before the LoRA low-rank projection.
    freeze_base_model:
        If True, freezes all original T5 parameters and trains only LoRA and
        gating parameters. If False, the base model remains trainable.
    target_decoder_modules:
        Name suffixes of decoder linear modules to wrap. By default, we wrap
        decoder self-attention projections and decoder feed-forward projections.
        This follows the view-aware decoding design while avoiding encoder-side
        changes.
    gate_loss_weight:
        Weight of the auxiliary gating supervision loss.
    contrastive_loss_weight:
        Weight of the optional group-wise contrastive loss.
    """

    DEFAULT_TARGETS: Tuple[str, ...] = (
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
        "DenseReluDense.wi",
        "DenseReluDense.wi_0",
        "DenseReluDense.wi_1",
        "DenseReluDense.wo",
    )

    def __init__(
        self,
        model_name: str = "t5-base",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        freeze_base_model: bool = False,
        target_decoder_modules: Optional[Iterable[str]] = None,
        gate_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model
        self.gate_loss_weight = gate_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        if freeze_base_model:
            for p in self.model.parameters():
                p.requires_grad = False

        self.target_decoder_modules = tuple(target_decoder_modules or self.DEFAULT_TARGETS)
        self.view_lora_modules: List[ViewAwareLoRALinear] = []
        self._inject_view_aware_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=freeze_base_model,
        )

    def _is_target_decoder_linear(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not name.startswith("decoder."):
            return False
        return any(name.endswith(suffix) for suffix in self.target_decoder_modules)

    def _get_parent_module(self, module_name: str) -> Tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]  # type: ignore[index]
            else:
                parent = getattr(parent, part)
        return parent, parts[-1]

    def _inject_view_aware_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool,
    ) -> None:
        # Collect names first because we mutate the module tree afterwards.
        target_names = [
            name for name, module in self.model.named_modules()
            if self._is_target_decoder_linear(name, module)
        ]

        for name in target_names:
            parent, child_name = self._get_parent_module(name)
            base_linear = getattr(parent, child_name)
            wrapped = ViewAwareLoRALinear(
                base_linear=base_linear,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                freeze_base=freeze_base,
            )
            setattr(parent, child_name, wrapped)
            self.view_lora_modules.append(wrapped)

        if len(self.view_lora_modules) == 0:
            raise RuntimeError(
                "No decoder linear modules were wrapped. Please check "
                "target_decoder_modules for the selected T5 checkpoint."
            )

    @staticmethod
    def groupwise_contrastive_loss(
        doc_rep: torch.Tensor,
        docid_reps: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Multi-positive contrastive loss for document and docid views.

        doc_rep: [B, D]
        docid_reps: [B, N, D], where N is the number of docids per document.
        """
        doc_rep = F.normalize(doc_rep, dim=-1)
        docid_reps = F.normalize(docid_reps, dim=-1)

        # sim[b, j, n] compares document b with docid n of document j.
        sim = torch.einsum("bd,jnd->bjn", doc_rep, docid_reps) / temperature
        exp_sim = torch.exp(sim)

        batch_size = doc_rep.size(0)
        pos_mask = torch.eye(batch_size, device=doc_rep.device, dtype=torch.bool)
        positives = exp_sim[pos_mask].view(batch_size, -1).sum(dim=-1)
        denominator = exp_sim.view(batch_size, -1).sum(dim=-1)
        loss = -torch.log((positives + 1e-8) / (denominator + 1e-8))
        return loss.mean()

    def _collect_gate_probs(self) -> Optional[torch.Tensor]:
        gate_probs = [m.last_gate_probs for m in self.view_lora_modules if m.last_gate_probs is not None]
        if len(gate_probs) == 0:
            return None

        # Different decoder modules can be called with the same [B, T, D] shape.
        # We average their gate distributions to obtain a stable auxiliary signal.
        stacked = torch.stack(gate_probs, dim=0)  # [M, B, T, 2]
        return stacked.mean(dim=0)

    def _compute_gate_loss(
        self,
        gate_probs: Optional[torch.Tensor],
        view_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute supervised gating loss.

        view_labels can be either:
        - [B], one label per instance; or
        - [B, T], one label per decoder token.

        Labels use 0 for global/title docids and 1 for local docids. Positions
        with value -100 are ignored.
        """
        if gate_probs is None or view_labels is None:
            return None

        if view_labels.dim() == 1:
            view_labels = view_labels[:, None].expand(-1, gate_probs.size(1))
        elif view_labels.dim() != 2:
            raise ValueError("view_labels must have shape [B] or [B, T].")

        # If labels are provided, ignore padded decoder positions.
        if labels is not None:
            max_len = min(view_labels.size(1), labels.size(1), gate_probs.size(1))
            view_labels = view_labels[:, :max_len]
            label_mask = labels[:, :max_len] != -100
            gate_probs = gate_probs[:, :max_len, :]
        else:
            label_mask = view_labels != -100

        valid_mask = (view_labels != -100) & label_mask
        if valid_mask.sum() == 0:
            return None

        log_probs = torch.log(gate_probs.clamp_min(1e-8))
        loss = F.nll_loss(
            log_probs[valid_mask],
            view_labels[valid_mask].long(),
            reduction="mean",
        )
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        view_labels: Optional[torch.Tensor] = None,
        doc_rep: Optional[torch.Tensor] = None,
        docid_reps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CORALOutput:
        """Run training or scoring forward pass.

        view_labels supervises the adapter gate. Use 0 for global/title docids,
        1 for local docids, and -100 for ignored positions.
        """
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        gate_probs = self._collect_gate_probs()
        gate_loss = self._compute_gate_loss(gate_probs, view_labels, labels)

        contrastive_loss = None
        if doc_rep is not None and docid_reps is not None:
            contrastive_loss = self.groupwise_contrastive_loss(doc_rep, docid_reps)

        total_loss = outputs.loss
        if gate_loss is not None:
            total_loss = total_loss + self.gate_loss_weight * gate_loss
        if contrastive_loss is not None:
            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss

        return CORALOutput(
            loss=total_loss,
            logits=outputs.logits,
            lm_loss=outputs.loss,
            gate_loss=gate_loss,
            contrastive_loss=contrastive_loss,
            gate_probs=gate_probs,
        )

    @torch.no_grad()
    def generate_docids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_allowed_tokens_fn=None,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        max_new_tokens: int = 32,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate docids, optionally constrained by a prefix trie.

        Pass a HuggingFace-compatible prefix_allowed_tokens_fn(batch_id, input_ids)
        to enforce trie-constrained beam search.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            **generate_kwargs,
        )

    def trainable_parameter_summary(self) -> Dict[str, Union[int, float]]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
            "num_view_lora_modules": len(self.view_lora_modules),
        }
